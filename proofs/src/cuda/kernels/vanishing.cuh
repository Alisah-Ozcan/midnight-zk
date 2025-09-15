#ifndef VANISHING_KERNELS_CUH
#define VANISHING_KERNELS_CUH

// ADDITION POINTER POINTER
__global__ void mul_chunks(fr_t* input1, fr_t* input2, fr_t* output,
                           const size_t size, const size_t mask) {

    size_t tid = threadIdx.x + (size_t)blockIdx.x * blockDim.x;

    if (tid >= size) return;

    fr_t input1_local = input1[tid];
    fr_t input2_local = input2[tid & mask];

    input1_local = input1_local * input2_local;

    output[tid] = input1_local;
}


    // fn distribute_powers_zeta(&self, a: &mut [F], into_coset: bool) {
    //     let coset_powers = if into_coset {
    //         [self.g_coset, self.g_coset_inv]
    //     } else {
    //         [self.g_coset_inv, self.g_coset]
    //     };
    //     parallelize(a, |a, mut index| {
    //         for a in a {
    //             // Distribute powers to move into/from coset
    //             let i = index % (coset_powers.len() + 1);
    //             if i != 0 {
    //                 *a *= &coset_powers[i - 1];
    //             }
    //             index += 1;
    //         }
    //     });
    // }

__global__ void mul_inv_zeta(fr_t* input, fr_t* output,
                             const size_t size, fr_t zeta, fr_t zeta_inv) {

    size_t tid = threadIdx.x + (size_t)blockIdx.x * blockDim.x;

    if (tid >= size) return;

    if (tid % 3 == 0) {
        output[tid] = input[tid];
    } else if (tid % 3 == 1) {
        output[tid] = input[tid] * zeta_inv;
    } else { // tid % 3 == 2
        output[tid] = input[tid] * zeta;
    }

}




#endif