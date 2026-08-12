#include "add_metax.cuh"

#include <mcr/mc_runtime.h>
#include <common/maca_fp16.h>
#include <common/maca_bfloat16.h>

namespace llaisys::ops::metax {

constexpr int BLOCK_SIZE = 256;

template <typename T>
__global__ void add_kernel(T *c, const T *a, const T *b, size_t size) {
    size_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < size) {
        c[i] = a[i] + b[i];
    }
}

void add(std::byte *c, const std::byte *a, const std::byte *b,
         llaisysDataType_t type, size_t size) {
    size_t grid = (size + BLOCK_SIZE - 1) / BLOCK_SIZE;
    switch (type) {
    case LLAISYS_DTYPE_F32:
        add_kernel<float><<<grid, BLOCK_SIZE>>>(
            (float *)c, (const float *)a, (const float *)b, size);
        break;
    case LLAISYS_DTYPE_F16:
        add_kernel<__half><<<grid, BLOCK_SIZE>>>(
            (__half *)c, (const __half *)a, (const __half *)b, size);
        break;
    case LLAISYS_DTYPE_BF16:
        add_kernel<maca_bfloat16><<<grid, BLOCK_SIZE>>>(
            (maca_bfloat16 *)c, (const maca_bfloat16 *)a,
            (maca_bfloat16 *)b, size);
        break;
    default:
        break;
    }
}

} // namespace llaisys::ops::metax
