#include "embedding_metax.cuh"

#include <mcr/mc_runtime.h>
#include <common/maca_fp16.h>
#include <common/maca_bfloat16.h>

namespace llaisys::ops::metax {

constexpr int BLOCK_SIZE = 256;

template <typename T>
__global__ void embedding_kernel(T *out, const int64_t *index,
                                  const T *weight, size_t seq_len, size_t dim) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    size_t total = seq_len * dim;
    if (idx >= total) return;

    size_t row = idx / dim;
    size_t col = idx % dim;
    int64_t token = index[row];
    out[idx] = weight[token * dim + col];
}

void embedding(std::byte *out, const std::byte *index, const std::byte *weight,
               llaisysDataType_t dtype, size_t seq_len, size_t dim) {
    size_t total = seq_len * dim;
    size_t grid = (total + BLOCK_SIZE - 1) / BLOCK_SIZE;
    switch (dtype) {
    case LLAISYS_DTYPE_F32:
        embedding_kernel<float><<<grid, BLOCK_SIZE>>>(
            (float *)out, (const int64_t *)index, (const float *)weight,
            seq_len, dim);
        break;
    case LLAISYS_DTYPE_F16:
        embedding_kernel<__half><<<grid, BLOCK_SIZE>>>(
            (__half *)out, (const int64_t *)index, (const __half *)weight,
            seq_len, dim);
        break;
    case LLAISYS_DTYPE_BF16:
        embedding_kernel<maca_bfloat16><<<grid, BLOCK_SIZE>>>(
            (maca_bfloat16 *)out, (const int64_t *)index,
            (const maca_bfloat16 *)weight, seq_len, dim);
        break;
    default:
        break;
    }
}

} // namespace llaisys::ops::metax
