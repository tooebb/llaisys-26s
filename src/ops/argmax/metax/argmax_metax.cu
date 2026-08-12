#include "argmax_metax.cuh"

#include <mcr/mc_runtime.h>
#include <common/maca_fp16.h>
#include <common/maca_bfloat16.h>

#include <cfloat>

namespace llaisys::ops::metax {

constexpr int BLOCK_SIZE = 256;

template <typename T>
__global__ void argmax_reduce_kernel(const T *vals, size_t numel,
                                      int64_t *partial_max_idx,
                                      T *partial_max_val) {
    __shared__ T s_vals[BLOCK_SIZE];
    __shared__ int64_t s_idx[BLOCK_SIZE];

    size_t tid = threadIdx.x;
    size_t i = blockIdx.x * blockDim.x + tid;

    T max_val = -T(FLT_MAX);
    int64_t max_idx = -1;

    for (size_t j = i; j < numel; j += gridDim.x * blockDim.x) {
        if (vals[j] > max_val) {
            max_val = vals[j];
            max_idx = (int64_t)j;
        }
    }

    s_vals[tid] = max_val;
    s_idx[tid] = max_idx;
    __syncthreads();

    for (size_t s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            if (s_vals[tid + s] > s_vals[tid]) {
                s_vals[tid] = s_vals[tid + s];
                s_idx[tid] = s_idx[tid + s];
            }
        }
        __syncthreads();
    }

    if (tid == 0) {
        partial_max_idx[blockIdx.x] = s_idx[0];
        partial_max_val[blockIdx.x] = s_vals[0];
    }
}

template <typename T>
void launch_argmax(std::byte *max_idx, std::byte *max_val,
                   const std::byte *vals, size_t numel) {
    size_t grid = (numel + BLOCK_SIZE - 1) / BLOCK_SIZE;
    if (grid > 1024) grid = 1024;

    int64_t *d_partial_idx;
    T *d_partial_val;
    mcMalloc(&d_partial_idx, grid * sizeof(int64_t));
    mcMalloc(&d_partial_val, grid * sizeof(T));

    argmax_reduce_kernel<T><<<grid, BLOCK_SIZE>>>(
        (const T *)vals, numel, d_partial_idx, d_partial_val);

    argmax_reduce_kernel<T><<<1, BLOCK_SIZE>>>(
        d_partial_val, grid, (int64_t *)max_idx, (T *)max_val);

    mcFree(d_partial_idx);
    mcFree(d_partial_val);
}

void argmax(std::byte *max_idx, std::byte *max_val, const std::byte *vals,
            llaisysDataType_t dtype, size_t numel) {
    switch (dtype) {
    case LLAISYS_DTYPE_F32:
        launch_argmax<float>(max_idx, max_val, vals, numel);
        break;
    case LLAISYS_DTYPE_F16:
        launch_argmax<__half>(max_idx, max_val, vals, numel);
        break;
    case LLAISYS_DTYPE_BF16:
        launch_argmax<maca_bfloat16>(max_idx, max_val, vals, numel);
        break;
    default:
        break;
    }
}

} // namespace llaisys::ops::metax
