#include "rms_norm_cpu.hpp"

#include "../../../utils.hpp"

#include <cmath>

namespace llaisys::ops::cpu {

template <typename T>
static void rms_norm_(T *out, const T *in, const T *weight,
                      size_t rows, size_t cols, float eps) {
    for (size_t row = 0; row < rows; row++) {
        /* 步骤1: 求该行所有元素平方的均值 */
        float sum_sq = 0.0f;
        for (size_t col = 0; col < cols; col++) {
            float val = llaisys::utils::cast<float>(in[row * cols + col]);
            sum_sq += val * val;
        }
        float mean_sq = sum_sq / (float)cols;

        /* 步骤2: 算 rsqrt */
        float rms = 1.0f / std::sqrt(mean_sq + eps);

        /* 步骤3: 归一化并乘权重 */
        for (size_t col = 0; col < cols; col++) {
            float val = llaisys::utils::cast<float>(in[row * cols + col]);
            float w = llaisys::utils::cast<float>(weight[col]);
            out[row * cols + col] = llaisys::utils::cast<T>(val * rms * w);
        }
    }
}

void rms_norm(std::byte *out, const std::byte *in, const std::byte *weight,
              llaisysDataType_t dtype, size_t rows, size_t cols, float eps) {

    switch (dtype) {

    case LLAISYS_DTYPE_F32: {
        float *out_ptr = reinterpret_cast<float *>(out);
        const float *in_ptr = reinterpret_cast<const float *>(in);
        const float *weight_ptr = reinterpret_cast<const float *>(weight);
        rms_norm_<float>(out_ptr, in_ptr, weight_ptr, rows, cols, eps);
        break;
    }

    case LLAISYS_DTYPE_F16: {
        llaisys::fp16_t *out_ptr = reinterpret_cast<llaisys::fp16_t *>(out);
        const llaisys::fp16_t *in_ptr = reinterpret_cast<const llaisys::fp16_t *>(in);
        const llaisys::fp16_t *weight_ptr = reinterpret_cast<const llaisys::fp16_t *>(weight);
        rms_norm_<llaisys::fp16_t>(out_ptr, in_ptr, weight_ptr, rows, cols, eps);
        break;
    }

    case LLAISYS_DTYPE_BF16: {
        llaisys::bf16_t *out_ptr = reinterpret_cast<llaisys::bf16_t *>(out);
        const llaisys::bf16_t *in_ptr = reinterpret_cast<const llaisys::bf16_t *>(in);
        const llaisys::bf16_t *weight_ptr = reinterpret_cast<const llaisys::bf16_t *>(weight);
        rms_norm_<llaisys::bf16_t>(out_ptr, in_ptr, weight_ptr, rows, cols, eps);
        break;
    }

    default:
        EXCEPTION_UNSUPPORTED_DATATYPE(dtype);
    }
}

} // namespace llaisys::ops::cpu
