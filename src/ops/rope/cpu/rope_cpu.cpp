#include "rope_cpu.hpp"

#include "../../../utils.hpp"

#include <cmath>
#include <cstdint>

namespace llaisys::ops::cpu {

template <typename T>
static void rope_(T *out, const T *in, const int64_t *pos_ids,
                  size_t seq_len, size_t n_heads, size_t dim, float theta) {
    size_t head_dim = dim;
    size_t half_dim = dim / 2;
    size_t head_stride = n_heads * head_dim;

    for (size_t a = 0; a < seq_len; a++) {
        int64_t p = pos_ids[a];

        for (size_t j = 0; j < half_dim; j++) {
            float freq = (float)p / std::pow(theta, 2.0f * (float)j / (float)head_dim);
            float cos_v = std::cos(freq);
            float sin_v = std::sin(freq);

            for (size_t b = 0; b < n_heads; b++) {
                size_t base = a * head_stride + b * head_dim;

                float x_a = llaisys::utils::cast<float>(in[base + j]);
                float x_b = llaisys::utils::cast<float>(in[base + j + half_dim]);

                out[base + j] = llaisys::utils::cast<T>(x_a * cos_v - x_b * sin_v);
                out[base + j + half_dim] = llaisys::utils::cast<T>(x_b * cos_v + x_a * sin_v);
            }
        }
    }
}

void rope(std::byte *out, const std::byte *in, const std::byte *pos_ids,
          llaisysDataType_t dtype, size_t seq_len, size_t n_heads,
          size_t dim, float theta) {

    switch (dtype) {

    case LLAISYS_DTYPE_F32: {
        float *out_ptr = reinterpret_cast<float *>(out);
        const float *in_ptr = reinterpret_cast<const float *>(in);
        const int64_t *pos_ptr = reinterpret_cast<const int64_t *>(pos_ids);
        rope_<float>(out_ptr, in_ptr, pos_ptr, seq_len, n_heads, dim, theta);
        break;
    }

    case LLAISYS_DTYPE_F16: {
        llaisys::fp16_t *out_ptr = reinterpret_cast<llaisys::fp16_t *>(out);
        const llaisys::fp16_t *in_ptr = reinterpret_cast<const llaisys::fp16_t *>(in);
        const int64_t *pos_ptr = reinterpret_cast<const int64_t *>(pos_ids);
        rope_<llaisys::fp16_t>(out_ptr, in_ptr, pos_ptr, seq_len, n_heads, dim, theta);
        break;
    }

    case LLAISYS_DTYPE_BF16: {
        llaisys::bf16_t *out_ptr = reinterpret_cast<llaisys::bf16_t *>(out);
        const llaisys::bf16_t *in_ptr = reinterpret_cast<const llaisys::bf16_t *>(in);
        const int64_t *pos_ptr = reinterpret_cast<const int64_t *>(pos_ids);
        rope_<llaisys::bf16_t>(out_ptr, in_ptr, pos_ptr, seq_len, n_heads, dim, theta);
        break;
    }

    default:
        EXCEPTION_UNSUPPORTED_DATATYPE(dtype);
    }
}

} // namespace llaisys::ops::cpu
