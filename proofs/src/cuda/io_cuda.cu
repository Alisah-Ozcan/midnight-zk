#include <cuda.h>

// Work around name clash between SPPARK's global `fmt(...)` helper function and
// the {fmt} library namespace that RMM headers include. We rename SPPARK's
// function to `sppark_fmt` for this translation unit, but temporarily undefine
// the macro while including RMM to avoid interfering with {fmt} headers.
#ifndef SPPARK_FMT_RENAMED
#define SPPARK_FMT_RENAMED 1
#define fmt sppark_fmt
#endif

#if defined(FEATURE_BLS12_381)
#include <ff/bls12-381.hpp>
#else
#error "No FEATURE! It has to be BLS12-381."
#endif

#define SPPARK_DONT_INSTANTIATE_TEMPLATES
#include <cuda_runtime.h>
#include <curand_kernel.h>

#include <atomic>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <ec/jacobian_t.hpp>
#include <ec/xyzz_t.hpp>
#include <iostream>
#include <memory>
#include <msm/pippenger.cuh>
#include <new>
#include <ntt/ntt.cuh>
#include <vector>

typedef jacobian_t<fp_t> point_t;
typedef xyzz_t<fp_t> bucket_t;
typedef bucket_t::affine_t affine_t;
typedef fr_t scalar_t;
typedef uint64_t gpu_addr_t;

#include "kernels/evaluate_h.cuh"
#include "kernels/prefix_product.cuh"
#include "kernels/rand_msm_helper.cuh"
// RMM device vector utilities
// Temporarily undefine our `fmt` macro while including RMM (which brings in {fmt}).
#pragma push_macro("fmt")
#undef fmt
#include <rmm/cuda_stream_view.hpp>
#include <rmm/device_uvector.hpp>
#include <rmm/mr/device/cuda_memory_resource.hpp>
#include <rmm/mr/device/per_device_resource.hpp>
#include <rmm/mr/device/pool_memory_resource.hpp>
#include <rmm/mr/device/statistics_resource_adaptor.hpp>
#pragma pop_macro("fmt")

// ------------------------------- POOL ------------------------------- //

class DeviceMemoryPool {
    using CudaResource = rmm::mr::cuda_memory_resource;
    using DevicePoolResource = rmm::mr::pool_memory_resource<CudaResource>;
    using DeviceStatsAdaptor =
        rmm::mr::statistics_resource_adaptor<rmm::mr::device_memory_resource>;

   public:
    static DeviceMemoryPool& instance() {
        static DeviceMemoryPool inst;
        return inst;
    }

    void initialize(int device_index = 0) {
        std::call_once(init_flag_, [&] {
            device_id_ = device_index;

            int cur = -1;
            if (cudaGetDevice(&cur) == cudaSuccess && cur != device_id_) {
                cudaSetDevice(device_id_);
            }

            device_base_ = std::make_unique<CudaResource>();

            size_t free_b = 0, total_b = 0;
            cudaMemGetInfo(&free_b, &total_b);

            auto roundup_256 = [](size_t n) noexcept {
                return ((n + 255ull) / 256ull) * 256ull;
            };
            auto init_size = roundup_256(static_cast<size_t>(free_b * 0.90));
            auto max_size = roundup_256(static_cast<size_t>(free_b * 0.95));

            device_pool_ = std::make_unique<DevicePoolResource>(device_base_.get(),
                                                                init_size, max_size);
            device_stats_adaptor_ =
                std::make_unique<DeviceStatsAdaptor>(device_pool_.get());

            prev_global_ = rmm::mr::get_current_device_resource();
            rmm::mr::set_current_device_resource(device_stats_adaptor_.get());

            use_pool_.store(true, std::memory_order_release);
            initialized_.store(true, std::memory_order_release);
        });
    }

    void set_use_pool(bool enable, bool set_global = true) {
        initialize(device_id_ < 0 ? 0 : device_id_);
        std::lock_guard<std::mutex> lock(toggle_mtx_);
        use_pool_.store(enable, std::memory_order_release);
        if (!set_global) return;

        if (enable) {
            rmm::mr::set_current_device_resource(device_stats_adaptor_.get());
        } else {
            if (prev_global_)
                rmm::mr::set_current_device_resource(prev_global_);
            else
                rmm::mr::set_current_device_resource(device_base_.get());
        }
    }

    void* allocate(size_t size, cudaStream_t stream) {
        initialize(device_id_ < 0 ? 0 : device_id_);
        auto* res = current_resource();
        return res->allocate(size, rmm::cuda_stream_view{stream});
    }

    void deallocate(void* ptr, size_t size, cudaStream_t stream) {
        (void)stream;
        auto* res = current_resource();
        res->deallocate(ptr, size, rmm::cuda_stream_view{stream});
    }

    rmm::mr::device_memory_resource* current_resource() {
        initialize(device_id_ < 0 ? 0 : device_id_);
        return use_pool_.load(std::memory_order_acquire)
                   ? static_cast<rmm::mr::device_memory_resource*>(
                         device_stats_adaptor_.get())
                   : static_cast<rmm::mr::device_memory_resource*>(device_base_.get());
    }

    ~DeviceMemoryPool() { clean_pool(); }

   private:
    DeviceMemoryPool() = default;
    DeviceMemoryPool(const DeviceMemoryPool&) = delete;
    DeviceMemoryPool& operator=(const DeviceMemoryPool&) = delete;

    void clean_pool() {
        if (!initialized_.load(std::memory_order_acquire)) return;

        if (prev_global_) {
            rmm::mr::set_current_device_resource(prev_global_);
            prev_global_ = nullptr;
        } else {
            rmm::mr::set_current_device_resource(device_base_.get());
        }

        device_stats_adaptor_.reset();
        device_pool_.reset();
        device_base_.reset();

        use_pool_.store(false, std::memory_order_release);
        initialized_.store(false, std::memory_order_release);
    }

   private:
    std::unique_ptr<CudaResource> device_base_{nullptr};
    std::unique_ptr<DevicePoolResource> device_pool_{nullptr};
    std::unique_ptr<DeviceStatsAdaptor> device_stats_adaptor_{nullptr};

    rmm::mr::device_memory_resource* prev_global_{nullptr};

    std::once_flag init_flag_;
    std::atomic<bool> use_pool_{false};
    std::atomic<bool> initialized_{false};
    std::mutex toggle_mtx_;
    int device_id_{-1};
};

extern "C" RustError::by_value gpu_set_device(size_t device_index) {
    const gpu_t& gpu = select_gpu();
    try {
        cudaSetDevice(static_cast<int>(device_index));
        CUDA_OK(cudaGetLastError());
    } catch (const cuda_error& e) {
        gpu.sync();
#ifdef TAKE_RESPONSIBILITY_FOR_ERROR_MESSAGE
        return RustError{e.code(), e.what()};
#else
        // return RustError{e.code()};
        return RustError{e.code(), strdup(e.what())};
#endif
    } catch (const std::exception& e) {
        gpu.sync();
        fprintf(stderr, "[STD] %s\n", e.what());
        return RustError{CUDA_ERROR_UNKNOWN, e.what()};
    } catch (...) {
        gpu.sync();
        fprintf(stderr, "Unknown C++ exception\n");
        return RustError{CUDA_ERROR_UNKNOWN};
    }

    return RustError{cudaSuccess};
}

extern "C" RustError::by_value gpu_memory_pool_initialize() {
    const gpu_t& gpu = select_gpu();
    try {
        DeviceMemoryPool::instance().initialize(0);
        CUDA_OK(cudaGetLastError());
    } catch (const cuda_error& e) {
        gpu.sync();
#ifdef TAKE_RESPONSIBILITY_FOR_ERROR_MESSAGE
        return RustError{e.code(), e.what()};
#else
        // return RustError{e.code()};
        return RustError{e.code(), strdup(e.what())};
#endif
    } catch (const std::exception& e) {
        gpu.sync();
        fprintf(stderr, "[STD] %s\n", e.what());
        return RustError{CUDA_ERROR_UNKNOWN, e.what()};
    } catch (...) {
        gpu.sync();
        fprintf(stderr, "Unknown C++ exception\n");
        return RustError{CUDA_ERROR_UNKNOWN};
    }

    return RustError{cudaSuccess};
}

extern "C" RustError::by_value gpu_memory_pool_enable() {
    const gpu_t& gpu = select_gpu();
    try {
        DeviceMemoryPool::instance().set_use_pool(true, true);
        CUDA_OK(cudaGetLastError());
    } catch (const cuda_error& e) {
        gpu.sync();
#ifdef TAKE_RESPONSIBILITY_FOR_ERROR_MESSAGE
        return RustError{e.code(), e.what()};
#else
        // return RustError{e.code()};
        return RustError{e.code(), strdup(e.what())};
#endif
    } catch (const std::exception& e) {
        gpu.sync();
        fprintf(stderr, "[STD] %s\n", e.what());
        return RustError{CUDA_ERROR_UNKNOWN, e.what()};
    } catch (...) {
        gpu.sync();
        fprintf(stderr, "Unknown C++ exception\n");
        return RustError{CUDA_ERROR_UNKNOWN};
    }

    return RustError{cudaSuccess};
}

extern "C" RustError::by_value gpu_memory_pool_disable() {
    const gpu_t& gpu = select_gpu();
    try {
        DeviceMemoryPool::instance().set_use_pool(false, true);
        CUDA_OK(cudaGetLastError());
    } catch (const cuda_error& e) {
        gpu.sync();
#ifdef TAKE_RESPONSIBILITY_FOR_ERROR_MESSAGE
        return RustError{e.code(), e.what()};
#else
        // return RustError{e.code()};
        return RustError{e.code(), strdup(e.what())};
#endif
    } catch (const std::exception& e) {
        gpu.sync();
        fprintf(stderr, "[STD] %s\n", e.what());
        return RustError{CUDA_ERROR_UNKNOWN, e.what()};
    } catch (...) {
        gpu.sync();
        fprintf(stderr, "Unknown C++ exception\n");
        return RustError{CUDA_ERROR_UNKNOWN};
    }

    return RustError{cudaSuccess};
}

extern "C" RustError::by_value gpu_memory_pool_allocate(size_t size,
                                                        gpu_addr_t* pointer) {
    const gpu_t& gpu = select_gpu();
    try {
        void* p = DeviceMemoryPool::instance().allocate(size, gpu);
        *pointer = reinterpret_cast<gpu_addr_t>(p);
        CUDA_OK(cudaGetLastError());
    } catch (const cuda_error& e) {
        gpu.sync();
#ifdef TAKE_RESPONSIBILITY_FOR_ERROR_MESSAGE
        return RustError{e.code(), e.what()};
#else
        // return RustError{e.code()};
        return RustError{e.code(), strdup(e.what())};
#endif
    } catch (const std::exception& e) {
        gpu.sync();
        fprintf(stderr, "[STD] %s\n", e.what());
        return RustError{CUDA_ERROR_UNKNOWN, e.what()};
    } catch (...) {
        gpu.sync();
        fprintf(stderr, "Unknown C++ exception\n");
        return RustError{CUDA_ERROR_UNKNOWN};
    }

    return RustError{cudaSuccess};
}

extern "C" RustError::by_value gpu_memory_pool_deallocate(size_t size,
                                                          gpu_addr_t* pointer) {
    const gpu_t& gpu = select_gpu();
    try {
        void* p = reinterpret_cast<void*>(*pointer);
        DeviceMemoryPool::instance().deallocate(p, size, gpu);
        *pointer = 0;
        CUDA_OK(cudaGetLastError());
    } catch (const cuda_error& e) {
        gpu.sync();
#ifdef TAKE_RESPONSIBILITY_FOR_ERROR_MESSAGE
        return RustError{e.code(), e.what()};
#else
        // return RustError{e.code()};
        return RustError{e.code(), strdup(e.what())};
#endif
    } catch (const std::exception& e) {
        gpu.sync();
        fprintf(stderr, "[STD] %s\n", e.what());
        return RustError{CUDA_ERROR_UNKNOWN, e.what()};
    } catch (...) {
        gpu.sync();
        fprintf(stderr, "Unknown C++ exception\n");
        return RustError{CUDA_ERROR_UNKNOWN};
    }

    return RustError{cudaSuccess};
}

extern "C" RustError::by_value gpu_memory_transfer_host_to_device(gpu_addr_t* dst,
                                                                  const void* src,
                                                                  size_t bytes) {
    const gpu_t& gpu = select_gpu();
    try {
        cudaMemcpyAsync(reinterpret_cast<void*>(*dst), src, bytes, cudaMemcpyHostToDevice,
                        gpu);
        CUDA_OK(cudaGetLastError());
    } catch (const cuda_error& e) {
        gpu.sync();
#ifdef TAKE_RESPONSIBILITY_FOR_ERROR_MESSAGE
        return RustError{e.code(), e.what()};
#else
        // return RustError{e.code()};
        return RustError{e.code(), strdup(e.what())};
#endif
    } catch (const std::exception& e) {
        gpu.sync();
        fprintf(stderr, "[STD] %s\n", e.what());
        return RustError{CUDA_ERROR_UNKNOWN, e.what()};
    } catch (...) {
        gpu.sync();
        fprintf(stderr, "Unknown C++ exception\n");
        return RustError{CUDA_ERROR_UNKNOWN};
    }

    return RustError{cudaSuccess};
}

extern "C" RustError::by_value gpu_memory_transfer_device_to_host(void* dst,
                                                                  const gpu_addr_t* src,
                                                                  size_t bytes) {
    const gpu_t& gpu = select_gpu();
    try {
        cudaMemcpyAsync(dst, reinterpret_cast<const void*>(*src), bytes,
                        cudaMemcpyDeviceToHost, gpu);
        CUDA_OK(cudaGetLastError());
    } catch (const cuda_error& e) {
        gpu.sync();
#ifdef TAKE_RESPONSIBILITY_FOR_ERROR_MESSAGE
        return RustError{e.code(), e.what()};
#else
        // return RustError{e.code()};
        return RustError{e.code(), strdup(e.what())};
#endif
    } catch (const std::exception& e) {
        gpu.sync();
        fprintf(stderr, "[STD] %s\n", e.what());
        return RustError{CUDA_ERROR_UNKNOWN, e.what()};
    } catch (...) {
        gpu.sync();
        fprintf(stderr, "Unknown C++ exception\n");
        return RustError{CUDA_ERROR_UNKNOWN};
    }

    return RustError{cudaSuccess};
}

extern "C" RustError::by_value gpu_memory_transfer_device_to_device(gpu_addr_t* dst,
                                                                    const gpu_addr_t* src,
                                                                    size_t bytes) {
    const gpu_t& gpu = select_gpu();
    try {
        cudaMemcpyAsync(reinterpret_cast<void*>(*dst),
                        reinterpret_cast<const void*>(*src), bytes,
                        cudaMemcpyDeviceToDevice, gpu);
        CUDA_OK(cudaGetLastError());
    } catch (const cuda_error& e) {
        gpu.sync();
#ifdef TAKE_RESPONSIBILITY_FOR_ERROR_MESSAGE
        return RustError{e.code(), e.what()};
#else
        // return RustError{e.code()};
        return RustError{e.code(), strdup(e.what())};
#endif
    } catch (const std::exception& e) {
        gpu.sync();
        fprintf(stderr, "[STD] %s\n", e.what());
        return RustError{CUDA_ERROR_UNKNOWN, e.what()};
    } catch (...) {
        gpu.sync();
        fprintf(stderr, "Unknown C++ exception\n");
        return RustError{CUDA_ERROR_UNKNOWN};
    }

    return RustError{cudaSuccess};
}

__global__ void zero_padding_coset_kernel(fr_t* input, fr_t* output, const fr_t g_coset,
                                          const fr_t g_coset_inv, int polysize) {
    const uint32_t tid = blockIdx.x * blockDim.x + threadIdx.x;

    if (tid < polysize) {
        fr_t input_r = input[tid];
        int index = tid % 3;
        if (index == 1) {
            input_r = input_r * g_coset;
        } else if (index == 2) {
            input_r = input_r * g_coset_inv;
        } else {
        }
        output[tid] = input_r;
    } else {
        fr_t value;
        value.zero();
        output[tid] = value;
    }
}

extern "C" RustError::by_value gpu_coset_ntt(gpu_addr_t* input_ptr, size_t input_len,
                                             gpu_addr_t* output_ptr, size_t output_len,
                                             const fr_t* g_coset,
                                             const fr_t* g_coset_inv) {
    const gpu_t& gpu = select_gpu();
    int threads_per_block = 512;
    try {
        uint32_t log_isize = uint32_t(log2(output_len));
        size_t grid_size =
            (int)((output_len + threads_per_block - 1) / threads_per_block);

        zero_padding_coset_kernel<<<grid_size, threads_per_block, 0, gpu>>>(
            reinterpret_cast<fr_t*>(*input_ptr), reinterpret_cast<fr_t*>(*output_ptr),
            *g_coset, *g_coset_inv, input_len);
        CUDA_OK(cudaGetLastError());

        NTT::Base_dev_ptr(gpu, reinterpret_cast<fr_t*>(*output_ptr), log_isize,
                          NTT::InputOutputOrder::NN, NTT::Direction::forward,
                          NTT::Type::standard);
        CUDA_OK(cudaGetLastError());
        gpu.sync();

    } catch (const cuda_error& e) {
        gpu.sync();
#ifdef TAKE_RESPONSIBILITY_FOR_ERROR_MESSAGE
        return RustError{e.code(), e.what()};
#else
        // return RustError{e.code()};
        return RustError{e.code(), strdup(e.what())};
#endif
    } catch (const std::exception& e) {
        gpu.sync();
        fprintf(stderr, "[STD] %s\n", e.what());
        return RustError{CUDA_ERROR_UNKNOWN, e.what()};
    } catch (...) {
        gpu.sync();
        fprintf(stderr, "Unknown C++ exception\n");
        return RustError{CUDA_ERROR_UNKNOWN};
    }

    return RustError{cudaSuccess};
}

extern "C" RustError::by_value gpu_coset_intt(gpu_addr_t* input_ptr, size_t input_len,
                                              gpu_addr_t* output_ptr, size_t output_len,
                                              const fr_t* g_coset,
                                              const fr_t* g_coset_inv) {
    const gpu_t& gpu = select_gpu();
    int threads_per_block = 512;
    try {
        uint32_t log_isize = uint32_t(log2(input_len));
        size_t grid_size = (int)((input_len + threads_per_block - 1) / threads_per_block);

        NTT::Base_dev_ptr(gpu, reinterpret_cast<fr_t*>(*input_ptr), log_isize,
                          NTT::InputOutputOrder::NN, NTT::Direction::inverse,
                          NTT::Type::standard);
        CUDA_OK(cudaGetLastError());

        zero_padding_coset_kernel<<<grid_size, threads_per_block, 0, gpu>>>(
            reinterpret_cast<fr_t*>(*input_ptr), reinterpret_cast<fr_t*>(*output_ptr),
            *g_coset_inv, *g_coset, input_len);
        CUDA_OK(cudaGetLastError());

        gpu.sync();

    } catch (const cuda_error& e) {
        gpu.sync();
#ifdef TAKE_RESPONSIBILITY_FOR_ERROR_MESSAGE
        return RustError{e.code(), e.what()};
#else
        // return RustError{e.code()};
        return RustError{e.code(), strdup(e.what())};
#endif
    } catch (const std::exception& e) {
        gpu.sync();
        fprintf(stderr, "[STD] %s\n", e.what());
        return RustError{CUDA_ERROR_UNKNOWN, e.what()};
    } catch (...) {
        gpu.sync();
        fprintf(stderr, "Unknown C++ exception\n");
        return RustError{CUDA_ERROR_UNKNOWN};
    }

    return RustError{cudaSuccess};
}

extern "C" RustError::by_value gpu_intt(gpu_addr_t* input_ptr, size_t input_len) {
    const gpu_t& gpu = select_gpu();
    try {
        uint32_t log_size = uint32_t(log2(input_len));
        NTT::Base_dev_ptr(gpu, reinterpret_cast<fr_t*>(*input_ptr), log_size,
                          NTT::InputOutputOrder::NN, NTT::Direction::inverse,
                          NTT::Type::standard);
        CUDA_OK(cudaGetLastError());
        gpu.sync();

    } catch (const cuda_error& e) {
        gpu.sync();
#ifdef TAKE_RESPONSIBILITY_FOR_ERROR_MESSAGE
        return RustError{e.code(), e.what()};
#else
        // return RustError{e.code()};
        return RustError{e.code(), strdup(e.what())};
#endif
    } catch (const std::exception& e) {
        gpu.sync();
        fprintf(stderr, "[STD] %s\n", e.what());
        return RustError{CUDA_ERROR_UNKNOWN, e.what()};
    } catch (...) {
        gpu.sync();
        fprintf(stderr, "Unknown C++ exception\n");
        return RustError{CUDA_ERROR_UNKNOWN};
    }

    return RustError{cudaSuccess};
}

class RandomMSM {
   public:
    enum class Group : int { G = 0, G_LAGRANGE = 1 };

    static RandomMSM& get_instance() {
        static thread_local RandomMSM instance;
        return instance;
    }

    scalar_t* get_randoms(const affine_t points[], size_t npoints, size_t ffi_affine_sz,
                          const gpu_t& gpu_s, Group group) {
        auto& data = groups_[static_cast<int>(group)];

        if (!data.initialized || data.size != npoints) {
            if (data.random_scalars) {
                DeviceMemoryPool::instance().deallocate(
                    data.random_scalars, data.size * sizeof(scalar_t), gpu_s);
                gpu_s.sync();
            }

            data.size = npoints;
            void* p = DeviceMemoryPool::instance().allocate(data.size * sizeof(scalar_t),
                                                            gpu_s);
            data.random_scalars = reinterpret_cast<scalar_t*>(p);
            CUDA_OK(cudaGetLastError());

            uint32_t loop = sizeof(scalar_t) / sizeof(uint32_t);
            uint64_t size_loop = data.size * loop;
            int num_sms = gpu_s.sm_count();
            const int threads_per_block = 512;
            int needed_blocks =
                int((size_loop + threads_per_block - 1) / threads_per_block);
            int blocks = std::min(needed_blocks, num_sms);

            uint32_t* raw_ptr = reinterpret_cast<uint32_t*>(data.random_scalars);
            random_scalars_kernel<<<blocks, threads_per_block, 0, gpu_s>>>(
                raw_ptr, size_loop, static_cast<uint32_t>(time(nullptr)));
            CUDA_OK(cudaGetLastError());

            size_t grid_size = (data.size + threads_per_block - 1) / threads_per_block;
            montgomery_conv<<<grid_size, threads_per_block, 0, gpu_s>>>(
                data.random_scalars, data.size);
            CUDA_OK(cudaGetLastError());

            msm_t<bucket_t, point_t, affine_t, scalar_t> msm{nullptr, npoints};
            msm.invoke(data.random_point, points, npoints, data.random_scalars, true,
                       ffi_affine_sz);

            data.initialized = true;
        }

        return data.random_scalars;
    }

    point_t get_point(Group group) const {
        const auto& data = groups_[static_cast<int>(group)];
        if (!data.initialized)
            throw std::runtime_error("Group " + std::to_string(int(group)) +
                                     " not initialized yet!");
        return data.random_point;
    }

    ~RandomMSM() {
        for (auto& data : groups_) {
            if (data.random_scalars) cudaFree(data.random_scalars);
        }
    }

    RandomMSM(const RandomMSM&) = delete;
    RandomMSM& operator=(const RandomMSM&) = delete;

   private:
    RandomMSM() = default;

    struct GroupData {
        point_t random_point;
        scalar_t* random_scalars = nullptr;
        uint64_t size = 0;
        bool initialized = false;
    };

    GroupData groups_[2];
};

extern "C" RustError::by_value gpu_msm(point_t* out1, point_t* out2,
                                       const affine_t points[], size_t npoints,
                                       gpu_addr_t* scalar_ptr, size_t ffi_affine_sz) {
    const gpu_t& gpu = select_gpu();

    try {
        scalar_t* d_scalars_pointer = reinterpret_cast<scalar_t*>(*scalar_ptr);

        scalar_t* d_rand_scalars_pointer = RandomMSM::get_instance().get_randoms(
            points, npoints, ffi_affine_sz, gpu, RandomMSM::Group::G);

        const int threads_per_block = 512;
        size_t grid_size = (int)((npoints + threads_per_block - 1) / threads_per_block);

        scalar_subtraction_kernel<<<grid_size, threads_per_block, 0, gpu>>>(
            d_scalars_pointer, d_rand_scalars_pointer, npoints);

        *out2 = RandomMSM::get_instance().get_point(RandomMSM::Group::G);

        msm_t<bucket_t, point_t, affine_t, scalar_t> msm{nullptr, npoints};
        msm.invoke(*out1, points, npoints, d_scalars_pointer, true, ffi_affine_sz);
        CUDA_OK(cudaGetLastError());

    } catch (const cuda_error& e) {
        gpu.sync();
#ifdef TAKE_RESPONSIBILITY_FOR_ERROR_MESSAGE
        return RustError{e.code(), e.what()};
#else
        return RustError{e.code()};
#endif
    }

    return RustError{cudaSuccess};
}

extern "C" RustError::by_value gpu_msm_lagrange(
    point_t* out1, point_t* out2, const affine_t points[], size_t npoints,
    gpu_addr_t* scalar_ptr, // const scalar_t scalars[],
    size_t ffi_affine_sz) {
    const gpu_t& gpu = select_gpu();

    try {
        scalar_t* d_scalars_pointer = reinterpret_cast<scalar_t*>(*scalar_ptr);

        scalar_t* d_rand_scalars_pointer = RandomMSM::get_instance().get_randoms(
            points, npoints, ffi_affine_sz, gpu, RandomMSM::Group::G_LAGRANGE);

        const int threads_per_block = 512;
        size_t grid_size = (int)((npoints + threads_per_block - 1) / threads_per_block);

        scalar_subtraction_kernel<<<grid_size, threads_per_block, 0, gpu>>>(
            d_scalars_pointer, d_rand_scalars_pointer, npoints);

        *out2 = RandomMSM::get_instance().get_point(RandomMSM::Group::G_LAGRANGE);

        msm_t<bucket_t, point_t, affine_t, scalar_t> msm{nullptr, npoints};
        msm.invoke(*out1, points, npoints, d_scalars_pointer, true, ffi_affine_sz);
        CUDA_OK(cudaGetLastError());

    } catch (const cuda_error& e) {
        gpu.sync();
#ifdef TAKE_RESPONSIBILITY_FOR_ERROR_MESSAGE
        return RustError{e.code(), e.what()};
#else
        return RustError{e.code()};
#endif
    }

    return RustError{cudaSuccess};
}

extern "C" {

enum class ValueSourceKind : uint8_t {
    Constant,
    Intermediate,
    Fixed,
    Advice,
    Instance,
    Challenge,
    Beta,
    Gamma,
    Theta,
    TrashChallenge,
    Y,
    PreviousValue
};

struct ValueSourceFFI {
    ValueSourceKind kind;
    size_t param0;
    size_t param1;
};

enum class CalculationKind : uint8_t {
    Add,
    Sub,
    Mul,
    Square,
    Double,
    Negate,
    Store,
    Horner
};

struct CalculationFFI {
    CalculationKind kind;
    ValueSourceFFI a;
    ValueSourceFFI b;
    ValueSourceFFI extra;
    const ValueSourceFFI* horner_parts_ptr;
    size_t horner_parts_len;
};

struct CalculationInfoFFI {
    CalculationFFI calculation;
    size_t target;
};

struct ResolvedInput {
    fr_t* pointer = nullptr;
    fr_t constant = fr_t{};
    bool is_constant = false;
};

} // extern "C"

ResolvedInput get_resolve_input(const ValueSourceFFI& src, fr_t constants,
                                fr_t* intermediates, const gpu_addr_t* fixed,
                                const gpu_addr_t* advice, const gpu_addr_t* instance,
                                fr_t challenges, fr_t beta, fr_t gamma, fr_t theta,
                                fr_t trash_challenge, fr_t y, fr_t* prev,
                                size_t chunk_offset) {
    switch (src.kind) {
        case ValueSourceKind::Constant: {
            return {nullptr, constants, true};
        }
        case ValueSourceKind::Intermediate: {
            return {intermediates + (src.param0 * chunk_offset), fr_t{}, false};
        }
        case ValueSourceKind::Fixed: {
            return {reinterpret_cast<fr_t*>(fixed[src.param0]), fr_t{}, false};
        }
        case ValueSourceKind::Advice: {
            return {reinterpret_cast<fr_t*>(advice[src.param0]), fr_t{}, false};
        }
        case ValueSourceKind::Instance: {
            return {reinterpret_cast<fr_t*>(instance[src.param0]), fr_t{}, false};
        }
        case ValueSourceKind::Challenge: {
            return {nullptr, challenges, true};
        }
        case ValueSourceKind::Beta: {
            return {nullptr, beta, true};
        }
        case ValueSourceKind::Gamma: {
            return {nullptr, gamma, true};
        }
        case ValueSourceKind::Theta: {
            return {nullptr, theta, true};
        }
        case ValueSourceKind::TrashChallenge: {
            return {nullptr, trash_challenge, true};
        }
        case ValueSourceKind::Y: {
            return {nullptr, y, true};
        }
        case ValueSourceKind::PreviousValue: {
            return {prev, fr_t{}, false};
        }
        default:
            throw std::invalid_argument("Unknown ValueSourceKind");
    }
}

extern "C" RustError::by_value custom_gates_evaluation(
    const CalculationInfoFFI* calculations, size_t calculations_count,
    const gpu_addr_t* fixed_ptrs, const gpu_addr_t* advice_ptrs,
    const gpu_addr_t* instance_ptrs, const fr_t* challenges, size_t challenges_ptr_len,
    const fr_t* beta, const fr_t* gamma, const fr_t* theta, const fr_t* trash_challenge,
    const fr_t* y, gpu_addr_t* output, const fr_t* constants, size_t constants_ptr_len,
    int* rotation_value, size_t rotation_ptr_len, int rot_scale, int poly_size

) {
    constexpr size_t CHUNK_SIZE = 1 << 18; // To minimize memory usage
    const gpu_t& gpu = select_gpu();

    size_t c_size = poly_size;
    size_t num_parts = 1;
    if (poly_size > CHUNK_SIZE) {
        c_size = CHUNK_SIZE;
        num_parts = poly_size / CHUNK_SIZE;
    }

    const int threads_per_block = 512;
    size_t grid_size = (int)((c_size + threads_per_block - 1) / threads_per_block);

    try {
        size_t total_intermediate_size = calculations_count * c_size;
        void* p = DeviceMemoryPool::instance().allocate(
            total_intermediate_size * sizeof(fr_t), gpu);
        fr_t* intermediate_device_ptrs = reinterpret_cast<fr_t*>(p);

        void* p2 = DeviceMemoryPool::instance().allocate(poly_size * sizeof(fr_t), gpu);
        fr_t* prev_device_ptrs = reinterpret_cast<fr_t*>(p2);
        cudaMemcpyAsync(prev_device_ptrs, reinterpret_cast<fr_t*>(*output),
                        poly_size * sizeof(fr_t), cudaMemcpyDeviceToDevice, gpu);

        std::vector<size_t> horner_index;
        const auto& calculations_in = calculations[calculations_count - 1].calculation;
        for (size_t i = 0; i < calculations_in.horner_parts_len; i++) {
            const auto& part = calculations_in.horner_parts_ptr[i];
            horner_index.push_back(part.param0);
        }

        void* p3 = DeviceMemoryPool::instance().allocate(
            horner_index.size() * sizeof(size_t), gpu);
        size_t* horner_index_device_ptrs = reinterpret_cast<size_t*>(p3);
        gpu.HtoD(horner_index_device_ptrs, horner_index.data(), horner_index.size());
        CUDA_OK(cudaGetLastError());
        gpu.sync();

        const fr_t zero_fr = fr_t{};
        auto const_at = [&](size_t idx, const fr_t* arr, size_t arr_len) -> fr_t {
            return (arr_len && idx < arr_len) ? arr[idx] : zero_fr;
        };

        for (size_t outer = 0; outer < num_parts; outer++) {
            for (size_t i = 0; i < calculations_count; ++i) {
                const auto& info = calculations[i];
                const auto& calc = info.calculation;

                switch (calc.kind) {
                    case CalculationKind::Add: {
                        ResolvedInput in_a = get_resolve_input(
                            calc.a, const_at(calc.a.param0, constants, constants_ptr_len),
                            intermediate_device_ptrs, fixed_ptrs, advice_ptrs,
                            instance_ptrs,
                            const_at(calc.a.param0, challenges, challenges_ptr_len),
                            *beta, *gamma, *theta, *trash_challenge, *y, prev_device_ptrs,
                            c_size);
                        ResolvedInput in_b = get_resolve_input(
                            calc.b, const_at(calc.b.param0, constants, constants_ptr_len),
                            intermediate_device_ptrs, fixed_ptrs, advice_ptrs,
                            instance_ptrs,
                            const_at(calc.b.param0, challenges, challenges_ptr_len),
                            *beta, *gamma, *theta, *trash_challenge, *y, prev_device_ptrs,
                            c_size);

                        if (!in_a.is_constant && !in_b.is_constant) {
                            fr_t* output_d =
                                intermediate_device_ptrs + (info.target * c_size);

                            add_pp_kernel<<<grid_size, threads_per_block, 0, gpu>>>(
                                in_a.pointer, in_b.pointer, output_d, c_size);
                            CUDA_OK(cudaGetLastError());
                        } else if (!in_a.is_constant && in_b.is_constant) {
                            fr_t* output_d =
                                intermediate_device_ptrs + (info.target * c_size);

                            add_pc_kernel<<<grid_size, threads_per_block, 0, gpu>>>(
                                in_a.pointer, in_b.constant, output_d, c_size);
                            CUDA_OK(cudaGetLastError());
                        } else if (in_a.is_constant && !in_b.is_constant) {
                            fr_t* output_d =
                                intermediate_device_ptrs + (info.target * c_size);

                            add_cp_kernel<<<grid_size, threads_per_block, 0, gpu>>>(
                                in_a.constant, in_b.pointer, output_d, c_size);
                            CUDA_OK(cudaGetLastError());
                        } else {
                            fr_t* output_d =
                                intermediate_device_ptrs + (info.target * c_size);

                            add_cc_kernel<<<grid_size, threads_per_block, 0, gpu>>>(
                                in_a.constant, in_b.constant, output_d, c_size);
                            CUDA_OK(cudaGetLastError());
                        }

                        break;
                    }
                    case CalculationKind::Sub: {
                        ResolvedInput in_a = get_resolve_input(
                            calc.a, const_at(calc.a.param0, constants, constants_ptr_len),
                            intermediate_device_ptrs, fixed_ptrs, advice_ptrs,
                            instance_ptrs,
                            const_at(calc.a.param0, challenges, challenges_ptr_len),
                            *beta, *gamma, *theta, *trash_challenge, *y, prev_device_ptrs,
                            c_size);
                        ResolvedInput in_b = get_resolve_input(
                            calc.b, const_at(calc.b.param0, constants, constants_ptr_len),
                            intermediate_device_ptrs, fixed_ptrs, advice_ptrs,
                            instance_ptrs,
                            const_at(calc.b.param0, challenges, challenges_ptr_len),
                            *beta, *gamma, *theta, *trash_challenge, *y, prev_device_ptrs,
                            c_size);

                        if (!in_a.is_constant && !in_b.is_constant) {
                            fr_t* output_d =
                                intermediate_device_ptrs + (info.target * c_size);

                            sub_pp_kernel<<<grid_size, threads_per_block, 0, gpu>>>(
                                in_a.pointer, in_b.pointer, output_d, c_size);
                            CUDA_OK(cudaGetLastError());
                        } else if (!in_a.is_constant && in_b.is_constant) {
                            fr_t* output_d =
                                intermediate_device_ptrs + (info.target * c_size);

                            sub_pc_kernel<<<grid_size, threads_per_block, 0, gpu>>>(
                                in_a.pointer, in_b.constant, output_d, c_size);
                            CUDA_OK(cudaGetLastError());
                        } else if (in_a.is_constant && !in_b.is_constant) {
                            fr_t* output_d =
                                intermediate_device_ptrs + (info.target * c_size);

                            sub_cp_kernel<<<grid_size, threads_per_block, 0, gpu>>>(
                                in_a.constant, in_b.pointer, output_d, c_size);
                            CUDA_OK(cudaGetLastError());
                        } else {
                            fr_t* output_d =
                                intermediate_device_ptrs + (info.target * c_size);

                            sub_cc_kernel<<<grid_size, threads_per_block, 0, gpu>>>(
                                in_a.constant, in_b.constant, output_d, c_size);
                            CUDA_OK(cudaGetLastError());
                        }

                        break;
                    }
                    case CalculationKind::Mul: {
                        ResolvedInput in_b = get_resolve_input(
                            calc.b, const_at(calc.b.param0, constants, constants_ptr_len),
                            intermediate_device_ptrs, fixed_ptrs, advice_ptrs,
                            instance_ptrs,
                            const_at(calc.b.param0, challenges, challenges_ptr_len),
                            *beta, *gamma, *theta, *trash_challenge, *y, prev_device_ptrs,
                            c_size);

                        ResolvedInput in_a = get_resolve_input(
                            calc.a, const_at(calc.a.param0, constants, constants_ptr_len),
                            intermediate_device_ptrs, fixed_ptrs, advice_ptrs,
                            instance_ptrs,
                            const_at(calc.a.param0, challenges, challenges_ptr_len),
                            *beta, *gamma, *theta, *trash_challenge, *y, prev_device_ptrs,
                            c_size);

                        if (!in_a.is_constant && !in_b.is_constant) {
                            fr_t* output_d =
                                intermediate_device_ptrs + (info.target * c_size);

                            mul_pp_kernel<<<grid_size, threads_per_block, 0, gpu>>>(
                                in_a.pointer, in_b.pointer, output_d, c_size);
                            CUDA_OK(cudaGetLastError());
                        } else if (!in_a.is_constant && in_b.is_constant) {
                            fr_t* output_d =
                                intermediate_device_ptrs + (info.target * c_size);

                            mul_pc_kernel<<<grid_size, threads_per_block, 0, gpu>>>(
                                in_a.pointer, in_b.constant, output_d, c_size);
                            CUDA_OK(cudaGetLastError());
                        } else if (in_a.is_constant && !in_b.is_constant) {
                            fr_t* output_d =
                                intermediate_device_ptrs + (info.target * c_size);

                            mul_cp_kernel<<<grid_size, threads_per_block, 0, gpu>>>(
                                in_a.constant, in_b.pointer, output_d, c_size);
                            CUDA_OK(cudaGetLastError());
                        } else {
                            fr_t* output_d =
                                intermediate_device_ptrs + (info.target * c_size);

                            mul_cc_kernel<<<grid_size, threads_per_block, 0, gpu>>>(
                                in_a.constant, in_b.constant, output_d, c_size);
                            CUDA_OK(cudaGetLastError());
                        }

                        break;
                    }
                    case CalculationKind::Square: {
                        ResolvedInput in_a = get_resolve_input(
                            calc.a, const_at(calc.a.param0, constants, constants_ptr_len),
                            intermediate_device_ptrs, fixed_ptrs, advice_ptrs,
                            instance_ptrs,
                            const_at(calc.a.param0, challenges, challenges_ptr_len),
                            *beta, *gamma, *theta, *trash_challenge, *y, prev_device_ptrs,
                            c_size);

                        if (!in_a.is_constant) {
                            fr_t* output_d =
                                intermediate_device_ptrs + (info.target * c_size);

                            square_p_kernel<<<grid_size, threads_per_block, 0, gpu>>>(
                                in_a.pointer, output_d, c_size);
                            CUDA_OK(cudaGetLastError());
                        } else {
                            fr_t* output_d =
                                intermediate_device_ptrs + (info.target * c_size);

                            square_c_kernel<<<grid_size, threads_per_block, 0, gpu>>>(
                                in_a.constant, output_d, c_size);
                            CUDA_OK(cudaGetLastError());
                        }

                        break;
                    }
                    case CalculationKind::Double: {
                        ResolvedInput in_a = get_resolve_input(
                            calc.a, const_at(calc.a.param0, constants, constants_ptr_len),
                            intermediate_device_ptrs, fixed_ptrs, advice_ptrs,
                            instance_ptrs,
                            const_at(calc.a.param0, challenges, challenges_ptr_len),
                            *beta, *gamma, *theta, *trash_challenge, *y, prev_device_ptrs,
                            c_size);

                        if (!in_a.is_constant) {
                            fr_t* output_d =
                                intermediate_device_ptrs + (info.target * c_size);

                            double_p_kernel<<<grid_size, threads_per_block, 0, gpu>>>(
                                in_a.pointer, output_d, c_size);
                            CUDA_OK(cudaGetLastError());
                        } else {
                            fr_t* output_d =
                                intermediate_device_ptrs + (info.target * c_size);

                            double_c_kernel<<<grid_size, threads_per_block, 0, gpu>>>(
                                in_a.constant, output_d, c_size);
                            CUDA_OK(cudaGetLastError());
                        }

                        break;
                    }
                    case CalculationKind::Negate: {
                        ResolvedInput in_a = get_resolve_input(
                            calc.a, const_at(calc.a.param0, constants, constants_ptr_len),
                            intermediate_device_ptrs, fixed_ptrs, advice_ptrs,
                            instance_ptrs,
                            const_at(calc.a.param0, challenges, challenges_ptr_len),
                            *beta, *gamma, *theta, *trash_challenge, *y, prev_device_ptrs,
                            c_size);

                        if (!in_a.is_constant) {
                            fr_t* output_d =
                                intermediate_device_ptrs + (info.target * c_size);

                            negate_p_kernel<<<grid_size, threads_per_block, 0, gpu>>>(
                                in_a.pointer, output_d, c_size);
                            CUDA_OK(cudaGetLastError());
                        } else {
                            fr_t* output_d =
                                intermediate_device_ptrs + (info.target * c_size);

                            negate_c_kernel<<<grid_size, threads_per_block, 0, gpu>>>(
                                in_a.constant, output_d, c_size);
                            CUDA_OK(cudaGetLastError());
                        }

                        break;
                    }
                    case CalculationKind::Store: {
                        ResolvedInput in_a = get_resolve_input(
                            calc.a, const_at(calc.a.param0, constants, constants_ptr_len),
                            intermediate_device_ptrs, fixed_ptrs, advice_ptrs,
                            instance_ptrs,
                            const_at(calc.a.param0, challenges, challenges_ptr_len),
                            *beta, *gamma, *theta, *trash_challenge, *y, prev_device_ptrs,
                            c_size);

                        fr_t* output_d =
                            intermediate_device_ptrs + (info.target * c_size);

                        int offset_in = outer * c_size;

                        store_kernel<<<grid_size, threads_per_block, 0, gpu>>>(
                            in_a.pointer, output_d, rotation_value[calc.a.param1],
                            rot_scale, poly_size, c_size, offset_in);
                        CUDA_OK(cudaGetLastError());

                        break;
                    }
                    case CalculationKind::Horner: {
                        ResolvedInput in_a = get_resolve_input(
                            calc.a, const_at(calc.a.param0, constants, constants_ptr_len),
                            intermediate_device_ptrs, fixed_ptrs, advice_ptrs,
                            instance_ptrs,
                            const_at(calc.a.param0, challenges, challenges_ptr_len),
                            *beta, *gamma, *theta, *trash_challenge, *y, prev_device_ptrs,
                            c_size);

                        // Intermediate
                        ResolvedInput in_c = get_resolve_input(
                            calc.extra,
                            (constants_ptr_len == 0 ? fr_t{}
                                                    : constants[calc.extra.param0]),
                            intermediate_device_ptrs, fixed_ptrs, advice_ptrs,
                            instance_ptrs,
                            (challenges_ptr_len == 0 ? fr_t{}
                                                     : challenges[calc.extra.param0]),
                            *beta, *gamma, *theta, *trash_challenge, *y, prev_device_ptrs,
                            c_size);

                        fr_t* output_d =
                            intermediate_device_ptrs + (info.target * c_size);

                        size_t horner_size = horner_index.size();

                        horner_kernel<<<grid_size, threads_per_block, 0, gpu>>>(
                            in_a.pointer + (outer * c_size), intermediate_device_ptrs,
                            output_d, in_c.constant, horner_index_device_ptrs,
                            horner_size, c_size);
                        CUDA_OK(cudaGetLastError());

                        break;
                    }
                    default:
                        throw std::invalid_argument("Unknown Calculation");
                }
            }

            int offset_in = outer * c_size;

            cudaMemcpyAsync(
                reinterpret_cast<fr_t*>(*output) + offset_in,
                intermediate_device_ptrs + ((calculations_count - 1) * c_size),
                c_size * sizeof(fr_t), cudaMemcpyDeviceToDevice, gpu);
        }

        DeviceMemoryPool::instance().deallocate(p, total_intermediate_size * sizeof(fr_t),
                                                gpu);
        DeviceMemoryPool::instance().deallocate(p2, poly_size * sizeof(fr_t), gpu);
        DeviceMemoryPool::instance().deallocate(p3, horner_index.size() * sizeof(size_t),
                                                gpu);

        CUDA_OK(cudaGetLastError());
        gpu.sync();

    } catch (const cuda_error& e) {
        gpu.sync();
#ifdef TAKE_RESPONSIBILITY_FOR_ERROR_MESSAGE
        return RustError{e.code(), e.what()};
#else
        // return RustError{e.code()};
        return RustError{e.code(), strdup(e.what())};
#endif
    } catch (const std::exception& e) {
        gpu.sync();
        fprintf(stderr, "[STD] %s\n", e.what());
        return RustError{CUDA_ERROR_UNKNOWN, e.what()};
    } catch (...) {
        gpu.sync();
        fprintf(stderr, "Unknown C++ exception\n");
        return RustError{CUDA_ERROR_UNKNOWN};
    }

    return RustError{cudaSuccess};
}

enum class AnyFFI : uint8_t { Advice, Fixed, Instance };

struct ColumnFFI {
    size_t index;
    AnyFFI column_type;
};

fr_t* get_any_input(const ColumnFFI& src, const gpu_addr_t* advice,
                    const gpu_addr_t* fixed, const gpu_addr_t* instance) {
    switch (src.column_type) {
        case AnyFFI::Advice: {
            return reinterpret_cast<fr_t*>(advice[src.index]);
        }
        case AnyFFI::Fixed: {
            return reinterpret_cast<fr_t*>(fixed[src.index]);
        }
        case AnyFFI::Instance: {
            return reinterpret_cast<fr_t*>(instance[src.index]);
        }
        default:
            throw std::invalid_argument("Unknown ValueSourceKind");
    }
}

__global__ void pow_mul_kernel(fr_t base, fr_t* __restrict__ out,
                               const fr_t delta_start) {
    const uint32_t tid = blockIdx.x * blockDim.x + threadIdx.x;

    fr_t delta_start_r = delta_start;
    fr_t base_r = base;
    uint32_t p = tid;

    fr_t sqr = base_r;
    base_r = fr_t::csel(base_r, fr_t::one(), p & 1);

#pragma unroll 1
    while (p >>= 1) {
        sqr *= sqr;
        if (p & 1) base_r *= sqr;
    }

    fr_t result = base_r * delta_start_r;
    out[tid] = result;
}

__global__ void mul_kernel(fr_t* value, const fr_t constant) {
    const uint32_t tid = blockIdx.x * blockDim.x + threadIdx.x;

    fr_t constant_r = constant;
    fr_t value_r = value[tid];

    value_r = value_r * constant_r;

    value[tid] = value_r;
}

__global__ void permutation_stage1_kernel1(fr_t* value, fr_t* first_perm, fr_t* last_perm,
                                           fr_t* l0, fr_t* l_last, fr_t y, int polysize) {
    const uint32_t tid = blockIdx.x * blockDim.x + threadIdx.x;

    if (tid >= polysize) return;

    const fr_t y_r = y;
    fr_t value_r = value[tid];
    const fr_t l0_r = l0[tid];
    fr_t first_perm_r = first_perm[tid];

    value_r = value_r * y_r;
    fr_t temp = fr_t::one() - first_perm_r;
    value_r = value_r + (temp * l0_r);

    fr_t last_perm_r = last_perm[tid];
    fr_t l_last_r = l_last[tid];

    value_r = value_r * y_r;
    temp = last_perm_r * last_perm_r;
    temp = temp - last_perm_r;
    temp = temp * l_last_r;
    value_r = value_r + temp;

    value[tid] = value_r;
}

__global__ void permutation_stage1_kernel2(fr_t* value, fr_t* perm1, fr_t* perm2,
                                           fr_t* l0, fr_t y, int rot, int rot_scale,
                                           int polysize) {
    const uint32_t tid = blockIdx.x * blockDim.x + threadIdx.x;

    if (tid >= polysize) return;

    const fr_t y_r = y;
    fr_t value_r = value[tid];
    const fr_t l0_r = l0[tid];

    size_t rotate_index = get_rotation_idx(tid, rot, rot_scale, polysize);

    fr_t perm1_r = perm1[tid];
    fr_t perm2_r = perm2[rotate_index];

    value_r = value_r * y_r;
    fr_t temp = perm1_r - perm2_r;
    temp = temp * l0_r;
    value_r = value_r + temp;

    value[tid] = value_r;
}

__global__ void permutation_stage2_left_kernel(fr_t* left, const fr_t* perm,
                                               const fr_t* column, const fr_t* coset,
                                               const fr_t beta, const fr_t gamma, int rot,
                                               int rot_scale, const int polysize,
                                               const bool first) {
    const uint32_t tid = blockIdx.x * blockDim.x + threadIdx.x;

    if (tid >= polysize) return;

    fr_t input;
    if (first) {
        size_t rotate_index = get_rotation_idx(tid, rot, rot_scale, polysize);
        input = perm[rotate_index];
    } else {
        input = left[tid];
    }

    fr_t column_r = column[tid];
    fr_t coset_r = coset[tid];
    fr_t beta_r = beta;
    fr_t gamma_r = gamma;

    fr_t temp = coset_r * beta_r;
    temp = temp + column_r;
    temp = temp + gamma_r;

    input = input * temp;

    left[tid] = input;
}

__global__ void permutation_stage2_right_kernel(fr_t* right, const fr_t* perm,
                                                const fr_t* column,
                                                const fr_t* current_delta,
                                                const fr_t gamma, const int polysize,
                                                const bool first) {
    const uint32_t tid = blockIdx.x * blockDim.x + threadIdx.x;

    if (tid >= polysize) return;

    fr_t input;
    if (first) {
        input = perm[tid];
    } else {
        input = right[tid];
    }

    fr_t column_r = column[tid];
    fr_t current_delta_r = current_delta[tid];
    fr_t gamma_r = gamma;

    fr_t temp = column_r + current_delta_r;
    temp = temp + gamma_r;

    input = input * temp;

    right[tid] = input;
}

__global__ void permutation_stage2_kernel(fr_t* value, const fr_t* left,
                                          const fr_t* right, const fr_t* l_active_row,
                                          const fr_t y, const int polysize) {
    const uint32_t tid = blockIdx.x * blockDim.x + threadIdx.x;

    if (tid >= polysize) return;

    const fr_t y_r = y;
    fr_t value_r = value[tid];

    fr_t left_r = left[tid];
    fr_t right_r = right[tid];
    fr_t l_active_row_r = l_active_row[tid];

    value_r = value_r * y;

    fr_t temp = left_r - right_r;
    temp = temp * l_active_row_r;

    value_r = value_r + temp;

    value[tid] = value_r;
}

extern "C" RustError::by_value permutations_evaluation(
    const ColumnFFI* column, size_t column_count,

    const gpu_addr_t* fixed_ptrs, const gpu_addr_t* advice_ptrs,
    const gpu_addr_t* instance_ptrs,

    gpu_addr_t* value,

    const gpu_addr_t* l0_ptrs, const gpu_addr_t* l_last_ptrs,
    const gpu_addr_t* l_active_row_ptrs, const gpu_addr_t* pk_coset_ptrs,

    const gpu_addr_t* permutation_ptrs, size_t permutation_ptr_len,

    const fr_t* delta_start, const fr_t* delta, const fr_t* beta, const fr_t* gamma,
    const fr_t* y, const fr_t* extended_omega, int chunk_len, int last_rotation_value,
    int rot_scale, int poly_size) {
    const gpu_t& gpu = select_gpu();
    const int threads_per_block = 256;
    size_t grid_size = (int)((poly_size + threads_per_block - 1) / threads_per_block);

    try {
        void* p = DeviceMemoryPool::instance().allocate(poly_size * sizeof(fr_t), gpu);
        fr_t* power_memory_ptrs = reinterpret_cast<fr_t*>(p);

        pow_mul_kernel<<<grid_size, threads_per_block, 0, gpu>>>(
            *extended_omega, power_memory_ptrs, *delta_start);
        CUDA_OK(cudaGetLastError());
        gpu.sync();

        fr_t* first_permutation_device_ptrs =
            reinterpret_cast<fr_t*>(permutation_ptrs[0]);
        fr_t* last_permutation_device_ptrs =
            reinterpret_cast<fr_t*>(permutation_ptrs[(permutation_ptr_len - 1)]);

        permutation_stage1_kernel1<<<grid_size, threads_per_block, 0, gpu>>>(
            reinterpret_cast<fr_t*>(*value), first_permutation_device_ptrs,
            last_permutation_device_ptrs, reinterpret_cast<fr_t*>(*l0_ptrs),
            reinterpret_cast<fr_t*>(*l_last_ptrs), *y, poly_size);
        CUDA_OK(cudaGetLastError());

        for (int i = 1; i < permutation_ptr_len; i++) {
            permutation_stage1_kernel2<<<grid_size, threads_per_block, 0, gpu>>>(
                reinterpret_cast<fr_t*>(*value),
                reinterpret_cast<fr_t*>(permutation_ptrs[i]),
                reinterpret_cast<fr_t*>(permutation_ptrs[i - 1]),
                reinterpret_cast<fr_t*>(*l0_ptrs), *y, last_rotation_value, rot_scale,
                poly_size);
            CUDA_OK(cudaGetLastError());
        }

        void* p2 =
            DeviceMemoryPool::instance().allocate(2 * poly_size * sizeof(fr_t), gpu);
        fr_t* left_ptrs = reinterpret_cast<fr_t*>(p2);
        fr_t* right_ptrs = left_ptrs + poly_size;

        size_t num_chunks = (column_count + chunk_len - 1) / chunk_len;
        assert(permutation_ptr_len == num_chunks);
        for (int i = 0; i < num_chunks; i++) {
            int begin = i * chunk_len;
            std::size_t end =
                std::min(static_cast<size_t>(begin + chunk_len), column_count);

            fr_t* perm_in_ptr = reinterpret_cast<fr_t*>(permutation_ptrs[i]);

            bool first_kernel = true;
            for (int j = 0; j < end - begin; j++) {
                fr_t* pointer_value = get_any_input(column[begin + j], advice_ptrs,
                                                    fixed_ptrs, instance_ptrs);

                fr_t* coset_ptrs = reinterpret_cast<fr_t*>(pk_coset_ptrs[(begin + j)]);

                permutation_stage2_left_kernel<<<grid_size, threads_per_block, 0, gpu>>>(
                    left_ptrs, perm_in_ptr, pointer_value, coset_ptrs, *beta, *gamma, 1,
                    rot_scale, poly_size, first_kernel);
                CUDA_OK(cudaGetLastError());

                first_kernel = false;
            }

            first_kernel = true;
            for (int j = 0; j < end - begin; j++) {
                fr_t* pointer_value = get_any_input(column[begin + j], advice_ptrs,
                                                    fixed_ptrs, instance_ptrs);

                permutation_stage2_right_kernel<<<grid_size, threads_per_block, 0, gpu>>>(
                    right_ptrs, perm_in_ptr, pointer_value, power_memory_ptrs, *gamma,
                    poly_size, first_kernel);
                CUDA_OK(cudaGetLastError());

                mul_kernel<<<grid_size, threads_per_block, 0, gpu>>>(power_memory_ptrs,
                                                                     *delta);
                CUDA_OK(cudaGetLastError());

                first_kernel = false;
            }

            permutation_stage2_kernel<<<grid_size, threads_per_block, 0, gpu>>>(
                reinterpret_cast<fr_t*>(*value), left_ptrs, right_ptrs,
                reinterpret_cast<fr_t*>(*l_active_row_ptrs), *y, poly_size);
            CUDA_OK(cudaGetLastError());
        }

        DeviceMemoryPool::instance().deallocate(p, poly_size * sizeof(fr_t), gpu);
        DeviceMemoryPool::instance().deallocate(p2, 2 * poly_size * sizeof(fr_t), gpu);

    } catch (const cuda_error& e) {
        gpu.sync();
#ifdef TAKE_RESPONSIBILITY_FOR_ERROR_MESSAGE
        return RustError{e.code(), e.what()};
#else
        // return RustError{e.code()};
        return RustError{e.code(), strdup(e.what())};
#endif
    } catch (const std::exception& e) {
        gpu.sync();
        fprintf(stderr, "[STD] %s\n", e.what());
        return RustError{CUDA_ERROR_UNKNOWN, e.what()};
    } catch (...) {
        gpu.sync();
        fprintf(stderr, "Unknown C++ exception\n");
        return RustError{CUDA_ERROR_UNKNOWN};
    }

    return RustError{cudaSuccess};
}

__global__ void lookups_stage1_kernel(fr_t* value, fr_t* product_coset, fr_t* l0,
                                      fr_t* l_last, fr_t y, int polysize) {
    const uint32_t tid = blockIdx.x * blockDim.x + threadIdx.x;

    if (tid >= polysize) return;

    const fr_t y_r = y;
    fr_t value_r = value[tid];
    fr_t product_coset_r = product_coset[tid];
    const fr_t l0_r = l0[tid];
    fr_t l_last_r = l_last[tid];

    value_r = value_r * y_r;
    fr_t temp = fr_t::one() - product_coset_r;
    value_r = value_r + (temp * l0_r);

    value_r = value_r * y_r;
    temp = product_coset_r * product_coset_r;
    temp = temp - product_coset_r;
    temp = temp * l_last_r;
    value_r = value_r + temp;

    value[tid] = value_r;
}

__global__ void lookups_stage2_kernel(fr_t* value, fr_t* table_value, fr_t* product_coset,
                                      fr_t* permuted_input_coset,
                                      fr_t* permuted_table_coset, fr_t* l_active_row,
                                      const fr_t y, const fr_t beta, const fr_t gamma,
                                      int rot_scale, int polysize) {
    const uint32_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= polysize) return;

    fr_t value_r = value[tid];
    value_r = value_r * y;

    fr_t temp1;
    {
        size_t r_next = get_rotation_idx(tid, 1, rot_scale, polysize);
        const fr_t product_coset_r_next = product_coset[r_next];
        const fr_t permuted_input_coset_r = permuted_input_coset[tid];
        temp1 = product_coset_r_next * (permuted_input_coset_r + beta);

        const fr_t permuted_table_coset_r = permuted_table_coset[tid];
        temp1 = temp1 * (permuted_table_coset_r + gamma);

        const fr_t product_coset_r = product_coset[tid];
        const fr_t table_value_r = table_value[tid];
        temp1 = temp1 - (product_coset_r * table_value_r);

        const fr_t l_active_row_r = l_active_row[tid];
        temp1 = temp1 * l_active_row_r;
    }

    value_r = value_r + temp1;
    value[tid] = value_r;
}

__global__ void lookups_stage3_kernel(fr_t* value, fr_t* permuted_input_coset,
                                      fr_t* permuted_table_coset, fr_t* l0,
                                      fr_t* l_active_row, const fr_t y, const fr_t beta,
                                      const fr_t gamma, int rot_scale, int polysize) {
    const uint32_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= polysize) return;

    fr_t value_r = value[tid];
    value_r = value_r * y;

    fr_t permuted_input_coset_r = permuted_input_coset[tid];
    fr_t permuted_table_coset_r = permuted_table_coset[tid];
    fr_t a_minus_s_r = permuted_input_coset_r - permuted_table_coset_r;

    fr_t l0_r = l0[tid];
    value_r = value_r + (a_minus_s_r * l0_r);

    value_r = value_r * y;

    size_t r_prev = get_rotation_idx(tid, -1, rot_scale, polysize);
    fr_t permuted_input_coset_r_prev = permuted_input_coset[r_prev];
    fr_t temp = a_minus_s_r * (permuted_input_coset_r - permuted_input_coset_r_prev);

    fr_t l_active_row_r = l_active_row[tid];
    temp = temp * l_active_row_r;
    value_r = value_r + temp;
    value[tid] = value_r;
}

extern "C" RustError::by_value lookups_evaluation(
    const CalculationInfoFFI* calculations, size_t calculations_count,

    const gpu_addr_t* fixed_ptrs, const gpu_addr_t* advice_ptrs,
    const gpu_addr_t* instance_ptrs,

    const gpu_addr_t* l0_ptrs, const gpu_addr_t* l_last_ptrs,
    const gpu_addr_t* l_active_row_ptrs,

    gpu_addr_t* value,

    const gpu_addr_t* product_coset_ptrs, const gpu_addr_t* permuted_input_coset,
    const gpu_addr_t* permuted_table_coset,

    const fr_t* challenges, size_t challenges_ptr_len, const fr_t* beta,
    const fr_t* gamma, const fr_t* theta, const fr_t* trash_challenge, const fr_t* y,
    const fr_t* constants, size_t constants_ptr_len, int* rotation_value,
    size_t rotation_ptr_len, int rot_scale, int poly_size) {
    constexpr size_t CHUNK_SIZE = 1 << 18; // To minimize memory usage
    const gpu_t& gpu = select_gpu();

    size_t c_size = poly_size;
    size_t num_parts = 1;
    if (poly_size > CHUNK_SIZE) {
        c_size = CHUNK_SIZE;
        num_parts = poly_size / CHUNK_SIZE;
    }

    const int threads_per_block = 512;
    size_t grid_size = (int)((c_size + threads_per_block - 1) / threads_per_block);

    try {
        void* p = DeviceMemoryPool::instance().allocate(poly_size * sizeof(fr_t), gpu);
        fr_t* table_value_device_ptrs = reinterpret_cast<fr_t*>(p);

        fr_t* prev_device_ptrs; // nullptr

        size_t total_intermediate_size = calculations_count * c_size;
        void* p2 = DeviceMemoryPool::instance().allocate(
            total_intermediate_size * sizeof(fr_t), gpu);
        fr_t* intermediate_device_ptrs = reinterpret_cast<fr_t*>(p2);

        const fr_t zero_fr = fr_t{};
        auto const_at = [&](size_t idx, const fr_t* arr, size_t arr_len) -> fr_t {
            return (arr_len && idx < arr_len) ? arr[idx] : zero_fr;
        };

        for (size_t outer = 0; outer < num_parts; outer++) {
            for (size_t i = 0; i < calculations_count; ++i) {
                const auto& info = calculations[i];
                const auto& calc = info.calculation;

                switch (calc.kind) {
                    case CalculationKind::Add: {
                        ResolvedInput in_a = get_resolve_input(
                            calc.a, const_at(calc.a.param0, constants, constants_ptr_len),
                            intermediate_device_ptrs, fixed_ptrs, advice_ptrs,
                            instance_ptrs,
                            const_at(calc.a.param0, challenges, challenges_ptr_len),
                            *beta, *gamma, *theta, *trash_challenge, *y, prev_device_ptrs,
                            c_size);
                        ResolvedInput in_b = get_resolve_input(
                            calc.b, const_at(calc.b.param0, constants, constants_ptr_len),
                            intermediate_device_ptrs, fixed_ptrs, advice_ptrs,
                            instance_ptrs,
                            const_at(calc.b.param0, challenges, challenges_ptr_len),
                            *beta, *gamma, *theta, *trash_challenge, *y, prev_device_ptrs,
                            c_size);

                        if (!in_a.is_constant && !in_b.is_constant) {
                            fr_t* output_d =
                                intermediate_device_ptrs + (info.target * c_size);

                            add_pp_kernel<<<grid_size, threads_per_block, 0, gpu>>>(
                                in_a.pointer, in_b.pointer, output_d, c_size);
                            CUDA_OK(cudaGetLastError());
                        } else if (!in_a.is_constant && in_b.is_constant) {
                            fr_t* output_d =
                                intermediate_device_ptrs + (info.target * c_size);

                            add_pc_kernel<<<grid_size, threads_per_block, 0, gpu>>>(
                                in_a.pointer, in_b.constant, output_d, c_size);
                            CUDA_OK(cudaGetLastError());
                        } else if (in_a.is_constant && !in_b.is_constant) {
                            fr_t* output_d =
                                intermediate_device_ptrs + (info.target * c_size);

                            add_cp_kernel<<<grid_size, threads_per_block, 0, gpu>>>(
                                in_a.constant, in_b.pointer, output_d, c_size);
                            CUDA_OK(cudaGetLastError());
                        } else {
                            fr_t* output_d =
                                intermediate_device_ptrs + (info.target * c_size);

                            add_cc_kernel<<<grid_size, threads_per_block, 0, gpu>>>(
                                in_a.constant, in_b.constant, output_d, c_size);
                            CUDA_OK(cudaGetLastError());
                        }

                        break;
                    }
                    case CalculationKind::Sub: {
                        ResolvedInput in_a = get_resolve_input(
                            calc.a, const_at(calc.a.param0, constants, constants_ptr_len),
                            intermediate_device_ptrs, fixed_ptrs, advice_ptrs,
                            instance_ptrs,
                            const_at(calc.a.param0, challenges, challenges_ptr_len),
                            *beta, *gamma, *theta, *trash_challenge, *y, prev_device_ptrs,
                            c_size);
                        ResolvedInput in_b = get_resolve_input(
                            calc.b, const_at(calc.b.param0, constants, constants_ptr_len),
                            intermediate_device_ptrs, fixed_ptrs, advice_ptrs,
                            instance_ptrs,
                            const_at(calc.b.param0, challenges, challenges_ptr_len),
                            *beta, *gamma, *theta, *trash_challenge, *y, prev_device_ptrs,
                            c_size);

                        if (!in_a.is_constant && !in_b.is_constant) {
                            fr_t* output_d =
                                intermediate_device_ptrs + (info.target * c_size);

                            sub_pp_kernel<<<grid_size, threads_per_block, 0, gpu>>>(
                                in_a.pointer, in_b.pointer, output_d, c_size);
                            CUDA_OK(cudaGetLastError());
                        } else if (!in_a.is_constant && in_b.is_constant) {
                            fr_t* output_d =
                                intermediate_device_ptrs + (info.target * c_size);

                            sub_pc_kernel<<<grid_size, threads_per_block, 0, gpu>>>(
                                in_a.pointer, in_b.constant, output_d, c_size);
                            CUDA_OK(cudaGetLastError());
                        } else if (in_a.is_constant && !in_b.is_constant) {
                            fr_t* output_d =
                                intermediate_device_ptrs + (info.target * c_size);

                            sub_cp_kernel<<<grid_size, threads_per_block, 0, gpu>>>(
                                in_a.constant, in_b.pointer, output_d, c_size);
                            CUDA_OK(cudaGetLastError());
                        } else {
                            fr_t* output_d =
                                intermediate_device_ptrs + (info.target * c_size);

                            sub_cc_kernel<<<grid_size, threads_per_block, 0, gpu>>>(
                                in_a.constant, in_b.constant, output_d, c_size);
                            CUDA_OK(cudaGetLastError());
                        }

                        break;
                    }
                    case CalculationKind::Mul: {
                        ResolvedInput in_b = get_resolve_input(
                            calc.b, const_at(calc.b.param0, constants, constants_ptr_len),
                            intermediate_device_ptrs, fixed_ptrs, advice_ptrs,
                            instance_ptrs,
                            const_at(calc.b.param0, challenges, challenges_ptr_len),
                            *beta, *gamma, *theta, *trash_challenge, *y, prev_device_ptrs,
                            c_size);

                        ResolvedInput in_a = get_resolve_input(
                            calc.a, const_at(calc.a.param0, constants, constants_ptr_len),
                            intermediate_device_ptrs, fixed_ptrs, advice_ptrs,
                            instance_ptrs,
                            const_at(calc.a.param0, challenges, challenges_ptr_len),
                            *beta, *gamma, *theta, *trash_challenge, *y, prev_device_ptrs,
                            c_size);

                        if (!in_a.is_constant && !in_b.is_constant) {
                            fr_t* output_d =
                                intermediate_device_ptrs + (info.target * c_size);

                            mul_pp_kernel<<<grid_size, threads_per_block, 0, gpu>>>(
                                in_a.pointer, in_b.pointer, output_d, c_size);
                            CUDA_OK(cudaGetLastError());
                        } else if (!in_a.is_constant && in_b.is_constant) {
                            fr_t* output_d =
                                intermediate_device_ptrs + (info.target * c_size);

                            mul_pc_kernel<<<grid_size, threads_per_block, 0, gpu>>>(
                                in_a.pointer, in_b.constant, output_d, c_size);
                            CUDA_OK(cudaGetLastError());
                        } else if (in_a.is_constant && !in_b.is_constant) {
                            fr_t* output_d =
                                intermediate_device_ptrs + (info.target * c_size);

                            mul_cp_kernel<<<grid_size, threads_per_block, 0, gpu>>>(
                                in_a.constant, in_b.pointer, output_d, c_size);
                            CUDA_OK(cudaGetLastError());
                        } else {
                            fr_t* output_d =
                                intermediate_device_ptrs + (info.target * c_size);

                            mul_cc_kernel<<<grid_size, threads_per_block, 0, gpu>>>(
                                in_a.constant, in_b.constant, output_d, c_size);
                            CUDA_OK(cudaGetLastError());
                        }

                        break;
                    }
                    case CalculationKind::Square: {
                        ResolvedInput in_a = get_resolve_input(
                            calc.a, const_at(calc.a.param0, constants, constants_ptr_len),
                            intermediate_device_ptrs, fixed_ptrs, advice_ptrs,
                            instance_ptrs,
                            const_at(calc.a.param0, challenges, challenges_ptr_len),
                            *beta, *gamma, *theta, *trash_challenge, *y, prev_device_ptrs,
                            c_size);

                        if (!in_a.is_constant) {
                            fr_t* output_d =
                                intermediate_device_ptrs + (info.target * c_size);

                            square_p_kernel<<<grid_size, threads_per_block, 0, gpu>>>(
                                in_a.pointer, output_d, c_size);
                            CUDA_OK(cudaGetLastError());
                        } else {
                            fr_t* output_d =
                                intermediate_device_ptrs + (info.target * c_size);

                            square_c_kernel<<<grid_size, threads_per_block, 0, gpu>>>(
                                in_a.constant, output_d, c_size);
                            CUDA_OK(cudaGetLastError());
                        }

                        break;
                    }
                    case CalculationKind::Double: {
                        ResolvedInput in_a = get_resolve_input(
                            calc.a, const_at(calc.a.param0, constants, constants_ptr_len),
                            intermediate_device_ptrs, fixed_ptrs, advice_ptrs,
                            instance_ptrs,
                            const_at(calc.a.param0, challenges, challenges_ptr_len),
                            *beta, *gamma, *theta, *trash_challenge, *y, prev_device_ptrs,
                            c_size);

                        if (!in_a.is_constant) {
                            fr_t* output_d =
                                intermediate_device_ptrs + (info.target * c_size);

                            double_p_kernel<<<grid_size, threads_per_block, 0, gpu>>>(
                                in_a.pointer, output_d, c_size);
                            CUDA_OK(cudaGetLastError());
                        } else {
                            fr_t* output_d =
                                intermediate_device_ptrs + (info.target * c_size);

                            double_c_kernel<<<grid_size, threads_per_block, 0, gpu>>>(
                                in_a.constant, output_d, c_size);
                            CUDA_OK(cudaGetLastError());
                        }

                        break;
                    }
                    case CalculationKind::Negate: {
                        ResolvedInput in_a = get_resolve_input(
                            calc.a, const_at(calc.a.param0, constants, constants_ptr_len),
                            intermediate_device_ptrs, fixed_ptrs, advice_ptrs,
                            instance_ptrs,
                            const_at(calc.a.param0, challenges, challenges_ptr_len),
                            *beta, *gamma, *theta, *trash_challenge, *y, prev_device_ptrs,
                            c_size);

                        if (!in_a.is_constant) {
                            fr_t* output_d =
                                intermediate_device_ptrs + (info.target * c_size);

                            negate_p_kernel<<<grid_size, threads_per_block, 0, gpu>>>(
                                in_a.pointer, output_d, c_size);
                            CUDA_OK(cudaGetLastError());
                        } else {
                            fr_t* output_d =
                                intermediate_device_ptrs + (info.target * c_size);

                            negate_c_kernel<<<grid_size, threads_per_block, 0, gpu>>>(
                                in_a.constant, output_d, c_size);
                            CUDA_OK(cudaGetLastError());
                        }

                        break;
                    }
                    case CalculationKind::Store: {
                        ResolvedInput in_a = get_resolve_input(
                            calc.a, const_at(calc.a.param0, constants, constants_ptr_len),
                            intermediate_device_ptrs, fixed_ptrs, advice_ptrs,
                            instance_ptrs,
                            const_at(calc.a.param0, challenges, challenges_ptr_len),
                            *beta, *gamma, *theta, *trash_challenge, *y, prev_device_ptrs,
                            c_size);

                        fr_t* output_d =
                            intermediate_device_ptrs + (info.target * c_size);

                        int offset_in = outer * c_size;

                        store_kernel<<<grid_size, threads_per_block, 0, gpu>>>(
                            in_a.pointer, output_d, rotation_value[calc.a.param1],
                            rot_scale, poly_size, c_size, offset_in);
                        CUDA_OK(cudaGetLastError());

                        break;
                    }
                    case CalculationKind::Horner: {
                        ResolvedInput in_a = get_resolve_input(
                            calc.a, const_at(calc.a.param0, constants, constants_ptr_len),
                            intermediate_device_ptrs, fixed_ptrs, advice_ptrs,
                            instance_ptrs,
                            const_at(calc.a.param0, challenges, challenges_ptr_len),
                            *beta, *gamma, *theta, *trash_challenge, *y, prev_device_ptrs,
                            c_size);

                        // Intermediate
                        ResolvedInput in_c = get_resolve_input(
                            calc.extra,
                            (constants_ptr_len == 0 ? fr_t{}
                                                    : constants[calc.extra.param0]),
                            intermediate_device_ptrs, fixed_ptrs, advice_ptrs,
                            instance_ptrs,
                            (challenges_ptr_len == 0 ? fr_t{}
                                                     : challenges[calc.extra.param0]),
                            *beta, *gamma, *theta, *trash_challenge, *y, prev_device_ptrs,
                            c_size);

                        fr_t* output_d =
                            intermediate_device_ptrs + (info.target * c_size);

                        std::vector<size_t> horner_index;
                        for (size_t i = 0; i < calc.horner_parts_len; i++) {
                            const auto& part = calc.horner_parts_ptr[i];
                            horner_index.push_back(part.param0);
                        }

                        void* p_in = DeviceMemoryPool::instance().allocate(
                            horner_index.size() * sizeof(size_t), gpu);
                        size_t* horner_index_device_ptrs =
                            reinterpret_cast<size_t*>(p_in);
                        gpu.HtoD(horner_index_device_ptrs, horner_index.data(),
                                 horner_index.size());
                        CUDA_OK(cudaGetLastError());

                        size_t horner_size = horner_index.size();

                        if (!in_a.is_constant) {
                            horner_kernel<<<grid_size, threads_per_block, 0, gpu>>>(
                                in_a.pointer + (outer * c_size), intermediate_device_ptrs,
                                output_d, in_c.constant, horner_index_device_ptrs,
                                horner_size, c_size);
                            CUDA_OK(cudaGetLastError());
                        } else {
                            horner_c_kernel<<<grid_size, threads_per_block, 0, gpu>>>(
                                in_a.constant, intermediate_device_ptrs, output_d,
                                in_c.constant, horner_index_device_ptrs, horner_size,
                                c_size);
                            CUDA_OK(cudaGetLastError());
                        }

                        DeviceMemoryPool::instance().deallocate(
                            p_in, horner_index.size() * sizeof(size_t), gpu);
                        break;
                    }
                    default:
                        throw std::invalid_argument("Unknown Calculation");
                }
            }

            int offset_in = outer * c_size;
            cudaMemcpyAsync(
                table_value_device_ptrs + offset_in,
                intermediate_device_ptrs + ((calculations_count - 1) * c_size),
                c_size * sizeof(fr_t), cudaMemcpyDeviceToDevice, gpu);
        }

        const int threads_per_block2 = 256;
        size_t grid_size2 =
            (int)((poly_size + threads_per_block2 - 1) / threads_per_block2);

        lookups_stage1_kernel<<<grid_size2, threads_per_block2, 0, gpu>>>(
            reinterpret_cast<fr_t*>(*value), reinterpret_cast<fr_t*>(*product_coset_ptrs),
            reinterpret_cast<fr_t*>(*l0_ptrs), reinterpret_cast<fr_t*>(*l_last_ptrs), *y,
            poly_size);
        CUDA_OK(cudaGetLastError());

        lookups_stage2_kernel<<<grid_size2, threads_per_block2, 0, gpu>>>(
            reinterpret_cast<fr_t*>(*value), table_value_device_ptrs,
            reinterpret_cast<fr_t*>(*product_coset_ptrs),
            reinterpret_cast<fr_t*>(*permuted_input_coset),
            reinterpret_cast<fr_t*>(*permuted_table_coset),
            reinterpret_cast<fr_t*>(*l_active_row_ptrs), *y, *beta, *gamma, rot_scale,
            poly_size);
        CUDA_OK(cudaGetLastError());

        lookups_stage3_kernel<<<grid_size2, threads_per_block2, 0, gpu>>>(
            reinterpret_cast<fr_t*>(*value),
            reinterpret_cast<fr_t*>(*permuted_input_coset),
            reinterpret_cast<fr_t*>(*permuted_table_coset),
            reinterpret_cast<fr_t*>(*l0_ptrs),
            reinterpret_cast<fr_t*>(*l_active_row_ptrs), *y, *beta, *gamma, rot_scale,
            poly_size);
        CUDA_OK(cudaGetLastError());

        DeviceMemoryPool::instance().deallocate(p, poly_size * sizeof(fr_t), gpu);
        DeviceMemoryPool::instance().deallocate(
            p2, total_intermediate_size * sizeof(fr_t), gpu);

    } catch (const cuda_error& e) {
        gpu.sync();
#ifdef TAKE_RESPONSIBILITY_FOR_ERROR_MESSAGE
        return RustError{e.code(), e.what()};
#else
        // return RustError{e.code()};
        return RustError{e.code(), strdup(e.what())};
#endif
    } catch (const std::exception& e) {
        gpu.sync();
        fprintf(stderr, "[STD] %s\n", e.what());
        return RustError{CUDA_ERROR_UNKNOWN, e.what()};
    } catch (...) {
        gpu.sync();
        fprintf(stderr, "Unknown C++ exception\n");
        return RustError{CUDA_ERROR_UNKNOWN};
    }

    return RustError{cudaSuccess};
}

__global__ void invert_kernel(const fr_t* input, fr_t* output, size_t size) {
    size_t tid = threadIdx.x + (size_t)blockIdx.x * blockDim.x;

    if (tid >= size) return;

    fr_t input1_r = input[tid];

    output[tid] = input1_r.reciprocal();
}

__global__ void commit_product_kernel1(const fr_t* permuted_input_value,
                                       const fr_t* permuted_table_value,
                                       fr_t* lookup_product, const fr_t beta,
                                       const fr_t gamma, size_t size) {
    size_t tid = threadIdx.x + (size_t)blockIdx.x * blockDim.x;

    if (tid >= size) return;

    const fr_t beta_r = beta;
    const fr_t gamma_r = gamma;

    fr_t permuted_input_value_r = permuted_input_value[tid];
    fr_t permuted_table_value_r = permuted_table_value[tid];

    fr_t temp = beta_r + permuted_input_value_r;
    temp = temp * (gamma_r + permuted_table_value_r);

    lookup_product[tid] = temp;
}

__global__ void commit_product_kernel2(fr_t* lookup_product,
                                       const fr_t* compressed_input_expression,
                                       const fr_t* compressed_table_expression,
                                       const fr_t beta, const fr_t gamma, size_t size) {
    size_t tid = threadIdx.x + (size_t)blockIdx.x * blockDim.x;

    if (tid >= size) return;

    const fr_t beta_r = beta;
    const fr_t gamma_r = gamma;

    fr_t temp = lookup_product[tid];
    fr_t compressed_input_expression_r = compressed_input_expression[tid];
    fr_t compressed_table_expression_r = compressed_table_expression[tid];

    temp = temp * (beta_r + compressed_input_expression_r);
    temp = temp * (gamma_r + compressed_table_expression_r);

    lookup_product[tid] = temp;
}

extern "C" RustError::by_value commit_product(
    const gpu_addr_t* permuted_input_value_device_ptrs,
    const gpu_addr_t* permuted_table_value_device_ptrs,
    const gpu_addr_t* compressed_input_expression_device_ptrs,
    const gpu_addr_t* compressed_table_expression_device_ptrs,

    gpu_addr_t* value,

    const fr_t* beta, const fr_t* gamma, const fr_t* randoms, const fr_t* one,
    int blinding_factors, int poly_size) {
    const gpu_t& gpu = select_gpu();
    assert(blinding_factors < poly_size);
    size_t small_size = poly_size - blinding_factors;

    const int threads_per_block = 512;
    size_t grid_size = (int)((poly_size + threads_per_block - 1) / threads_per_block);

    try {
        void* p =
            DeviceMemoryPool::instance().allocate((poly_size + 1) * sizeof(fr_t), gpu);
        fr_t* input_device_ptrs1 = reinterpret_cast<fr_t*>(p);
        fr_t* input_device_ptrs2 = input_device_ptrs1 + 1;

        commit_product_kernel1<<<grid_size, threads_per_block, 0, gpu>>>(
            reinterpret_cast<fr_t*>(*permuted_input_value_device_ptrs),
            reinterpret_cast<fr_t*>(*permuted_table_value_device_ptrs),
            input_device_ptrs2, *beta, *gamma, poly_size);
        CUDA_OK(cudaGetLastError());

        size_t grid_size_invert = (int)((poly_size + 256 - 1) / 256);
        invert_kernel<<<grid_size_invert, 256, 0, gpu>>>(input_device_ptrs2,
                                                         input_device_ptrs2, poly_size);

        commit_product_kernel2<<<grid_size, threads_per_block, 0, gpu>>>(
            input_device_ptrs2,
            reinterpret_cast<fr_t*>(*compressed_input_expression_device_ptrs),
            reinterpret_cast<fr_t*>(*compressed_table_expression_device_ptrs), *beta,
            *gamma, poly_size);
        CUDA_OK(cudaGetLastError());

        gpu.HtoD(input_device_ptrs1, one, 1);
        gpu.HtoD(reinterpret_cast<fr_t*>(*value) + small_size, randoms, blinding_factors);

        prefix_product_mod_inclusive(gpu, input_device_ptrs1,
                                     reinterpret_cast<fr_t*>(*value), small_size);

        DeviceMemoryPool::instance().deallocate(p, (poly_size + 1) * sizeof(fr_t), gpu);

        gpu.sync();
    } catch (const cuda_error& e) {
        gpu.sync();
#ifdef TAKE_RESPONSIBILITY_FOR_ERROR_MESSAGE
        return RustError{e.code(), e.what()};
#else
        // return RustError{e.code()};
        return RustError{e.code(), strdup(e.what())};
#endif
    } catch (const std::exception& e) {
        gpu.sync();
        fprintf(stderr, "[STD] %s\n", e.what());
        return RustError{CUDA_ERROR_UNKNOWN, e.what()};
    } catch (...) {
        gpu.sync();
        fprintf(stderr, "Unknown C++ exception\n");
        return RustError{CUDA_ERROR_UNKNOWN};
    }

    return RustError{cudaSuccess};
}