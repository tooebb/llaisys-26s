#include "rms_norm_metax.cuh"

#include <mcr/mc_runtime.h>
#include <common/maca_fp16.h>
#include <common/maca_bfloat16.h>

namespace llaisys::ops::metax {

constexpr int BLOCK_SIZE = 256;

template <typename T>
__global__ void rms_norm_kernel(T *out, const T *in, const T *weight,
                                 size_t rows, size_t cols, float eps) {
    size_t row = blockIdx.x;
    if (row >= rows) return;

    __shared__ float s_sum[BLOCK_SIZE];
    float local_sum = 0.0f;
    for (size_t j = threadIdx.x; j < cols; j += blockDim.x) {
        float v = float(in[row * cols + j]);
        local_sum += v * v;
    }

    s_sum[threadIdx.x] = local_sum;
    __syncthreads();
    for (size_t s = blockDim.x / 2; s > 0; s >>= 1) {
        if (threadIdx.x < s) {
            s_sum[threadIdx.x] += s_sum[threadIdx.x + s];
        }
        __syncthreads();
    }

    float rms = sqrtf(s_sum[0] / (float)cols + eps);

    for (size_t j = threadIdx.x; j < cols; j += blockDim.x) {
        float v = float(in[row * cols + j]);
        float w = float(weight[j]);
        out[row * cols + j] = T(v / rms * w);
    }
}

void rms_norm(std::byte *out, const std::byte *in, const std::byte *weight,
              llaisysDataType_t dtype, size_t rows, size_t cols, float eps) {
    switch (dtype) {
    case LLAISYS_DTYPE_F32:
        rms_norm_kernel<float><<<rows, BLOCK_SIZE>>>(
            (float *)out, (const float *)in, (const float *)weight,
            rows, cols, eps);
        break;
    case LLAISYS_DTYPE_F16:
        rms_norm_kernel<__half><<<rows, BLOCK_SIZE>>>(
            (__half *)out, (const __half *)in, (const __half *)weight,
            rows, cols, eps);
        break;
    case LLAISYS_DTYPE_BF16:
        rms_norm_kernel<maca_bfloat16><<<rows, BLOCK_SIZE>>>(
            (maca_bfloat16 *)out, (const maca_bfloat16 *)in,
            (maca_bfloat16 *)weight, rows, cols, eps);
        break;
    default:
        break;
    }
}

} // namespace llaisys::ops::metax
