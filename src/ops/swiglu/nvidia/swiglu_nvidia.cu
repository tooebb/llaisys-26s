#include "swiglu_nvidia.cuh"

namespace llaisys::ops::nvidia {

constexpr int BLOCK_SIZE = 256;

template <typename T>
__device__ T silu(T x) {
    return x / (T(1.0f) + expf(-float(x)));
}

template <typename T>
__global__ void swiglu_kernel(T *out, const T *gate, const T *up, size_t numel) {
    size_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < numel) {
        out[i] = silu(gate[i]) * up[i];
    }
}

void swiglu(std::byte *out, const std::byte *gate, const std::byte *up,
            llaisysDataType_t dtype, size_t numel) {
    size_t grid = (numel + BLOCK_SIZE - 1) / BLOCK_SIZE;
    switch (dtype) {
    case LLAISYS_DTYPE_F32:
        swiglu_kernel<float><<<grid, BLOCK_SIZE>>>(
            (float *)out, (const float *)gate, (const float *)up, numel);
        break;
    case LLAISYS_DTYPE_F16:
        swiglu_kernel<__half><<<grid, BLOCK_SIZE>>>(
            (__half *)out, (const __half *)gate, (const __half *)up, numel);
        break;
    case LLAISYS_DTYPE_BF16:
        swiglu_kernel<__nv_bfloat16><<<grid, BLOCK_SIZE>>>(
            (__nv_bfloat16 *)out, (const __nv_bfloat16 *)gate,
            (__nv_bfloat16 *)up, numel);
        break;
    default:
        break;
    }
}

} // namespace llaisys::ops::nvidia
