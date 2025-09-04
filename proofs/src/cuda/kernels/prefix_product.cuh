#ifndef PREFIX_PRODUCT_KERNELS_CUH
#define PREFIX_PRODUCT_KERNELS_CUH

// Hillis–Steele
template <int TILE>
__device__ void block_inclusive_scan_mul(fr_t* sh) {
    for (int offset = 1; offset < TILE; offset <<= 1) {
        int i = threadIdx.x;
        fr_t prev = (i >= offset) ? sh[i - offset] : fr_t::one();
        if (i >= offset) {
            sh[i] = sh[i] * prev;
        }
        __syncthreads();
    }
}

// Stage 1: Tile scan
template <int TILE>
__global__ void kernel_tile_scan_inclusive(const fr_t* __restrict__ X,
                                           fr_t* __restrict__ Z,
                                           fr_t* __restrict__ block_prod, int m) {
    __shared__ fr_t sh[TILE];
    int base = blockIdx.x * TILE;
    int tid = threadIdx.x;
    int idx = base + tid;

    fr_t v = fr_t::one();
    if (idx < m) v = X[idx];
    sh[tid] = v;
    __syncthreads();

    block_inclusive_scan_mul<TILE>(sh);

    if (idx < m) Z[idx] = sh[tid];

    if (tid == TILE - 1) {
        int remain = m - base;
        int last_valid = (remain <= 0 ? 0 : (remain >= TILE ? TILE - 1 : remain - 1));
        block_prod[blockIdx.x] = sh[last_valid];
    }
}

// Stage 2: block_prod over CHUNKED scan (shared)
template <int THREADS>
__global__ void scan_block_products_chunked_shared(fr_t* __restrict__ a, int B) {
    extern __shared__ fr_t sh[];
    __shared__ fr_t carry;
    __shared__ fr_t chunk_total;

    if (threadIdx.x == 0) carry = fr_t::one();
    __syncthreads();

    for (int start = 0; start < B; start += THREADS) {
        int tid = threadIdx.x;
        int count = ::min(THREADS, B - start);
        int idx = start + tid;

        fr_t x = (tid < count) ? a[idx] : fr_t::one();
        sh[tid] = x;
        __syncthreads();

        for (int offset = 1; offset < count; offset <<= 1) {
            fr_t prev = fr_t::one();
            if (tid < count && tid >= offset) prev = sh[tid - offset];
            if (tid < count && tid >= offset) {
                sh[tid] = sh[tid] * prev;
            }
            __syncthreads();
        }

        if (tid == count - 1) chunk_total = sh[tid];
        __syncthreads();

        if (tid < count) {
            a[idx] = carry * sh[tid];
        }

        if (tid == 0) {
            carry = carry * (count > 0 ? chunk_total : fr_t::one());
        }
    }
}

// Stage 3
template <int TILE>
__global__ void kernel_apply_block_prefix_mod(fr_t* __restrict__ Z,
                                              const fr_t* __restrict__ block_prod,
                                              int m) {
    int b = blockIdx.x;
    fr_t prefix = fr_t::one();
    if (b > 0) prefix = block_prod[b - 1];

    int base = b * TILE;
    int idx = base + threadIdx.x;
    if (idx < m) {
        Z[idx] = prefix * Z[idx];
    }
}

__host__ void prefix_product_mod_inclusive(const gpu_t& gpu, const fr_t* d_in,
                                           fr_t* d_out, int size) {
    constexpr int TILE = 256;
    constexpr int THREADS = 256;

    int m = size;
    int numBlocks = (m + TILE - 1) / TILE;
    int B = numBlocks;

    gpu_ptr_t<fr_t> devive_memory((fr_t*)gpu.Dmalloc(B * sizeof(fr_t)));
    CUDA_OK(cudaGetLastError());

    fr_t* dBlockProd = &devive_memory[0];

    kernel_tile_scan_inclusive<TILE><<<B, TILE, 0, gpu>>>(d_in, d_out, dBlockProd, m);
    CUDA_OK(cudaGetLastError());

    scan_block_products_chunked_shared<THREADS>
        <<<1, THREADS, THREADS * sizeof(fr_t), gpu>>>(dBlockProd, B);
    CUDA_OK(cudaGetLastError());

    kernel_apply_block_prefix_mod<TILE><<<B, TILE, 0, gpu>>>(d_out, dBlockProd, m);
    CUDA_OK(cudaGetLastError());
}
#endif // PREFIX_PRODUCT_KERNELS_CUH