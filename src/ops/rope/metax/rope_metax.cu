#include "rope_metax.cuh"

#include <mcr/mc_runtime.h>
#include <common/maca_fp16.h>
#include <common/maca_bfloat16.h>

namespace llaisys::ops::metax {

constexpr int BLOCK_SIZE = 256;

template <typename T>
__global__ void rope_kernel(T *out, const T *in, const int64_t *pos_ids,
                             size_t seq_len, size_t n_heads,
                             size_t dim, float theta) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    size_t total = seq_len * n_heads * (dim / 2);
    if (idx >= total) return;

    size_t half_dim = dim / 2;
    size_t half_idx = idx % half_dim;
    size_t head_idx = (idx / half_dim) % n_heads;
    size_t seq_idx = idx / (half_dim * n_heads);

    int64_t pos = pos_ids[seq_idx];
    float freq = 1.0f / powf(theta, (2.0f * half_idx) / (float)dim);
    float cos_val = cosf((float)pos * freq);
    float sin_val = sinf((float)pos * freq);

    size_t base = seq_idx * n_heads * dim + head_idx * dim;
    float x0 = float(in[base + half_idx]);
    float x1 = float(in[base + half_idx + half_dim]);

    out[base + half_idx]            = T(x0 * cos_val - x1 * sin_val);
    out[base + half_idx + half_dim] = T(x0 * sin_val + x1 * cos_val);
}

void rope(std::byte *out, const std::byte *in, const std::byte *pos_ids,
          llaisysDataType_t dtype, size_t seq_len, size_t n_heads,
          size_t dim, float theta) {
    size_t total = seq_len * n_heads * (dim / 2);
    size_t grid = (total + BLOCK_SIZE - 1) / BLOCK_SIZE;
    switch (dtype) {
    case LLAISYS_DTYPE_F32:
        rope_kernel<float><<<grid, BLOCK_SIZE>>>(
            (float *)out, (const float *)in, (const int64_t *)pos_ids,
            seq_len, n_heads, dim, theta);
        break;
    case LLAISYS_DTYPE_F16:
        rope_kernel<__half><<<grid, BLOCK_SIZE>>>(
            (__half *)out, (const __half *)in, (const int64_t *)pos_ids,
            seq_len, n_heads, dim, theta);
        break;
    case LLAISYS_DTYPE_BF16:
        rope_kernel<maca_bfloat16><<<grid, BLOCK_SIZE>>>(
            (maca_bfloat16 *)out, (const maca_bfloat16 *)in,
            (const int64_t *)pos_ids, seq_len, n_heads, dim, theta);
        break;
    default:
        break;
    }
}

} // namespace llaisys::ops::metax
