#include "linear_cpu.hpp"

#include "../../../utils.hpp"

namespace llaisys::ops::cpu {

template <typename T>
static void linear_(T *out, const T *in, const T *weight, const T *bias,
                    size_t M, size_t K, size_t N) {
    for (size_t i = 0; i < M; i++) {
        for (size_t j = 0; j < N; j++) {
            /* 用 float 累加，避免半精度溢出 */
            float sum = 0.0f;
            for (size_t k = 0; k < K; k++) {
                float a = llaisys::utils::cast<float>(in[i * K + k]);
                float b = llaisys::utils::cast<float>(weight[j * K + k]);
                sum += a * b;
            }
            if (bias != nullptr) {
                sum += llaisys::utils::cast<float>(bias[j]);
            }
            out[i * N + j] = llaisys::utils::cast<T>(sum);
        }
    }
}

void linear(std::byte *out, const std::byte *in, const std::byte *weight,
            const std::byte *bias, llaisysDataType_t dtype,
            size_t M, size_t K, size_t N) {

    switch (dtype) {

    case LLAISYS_DTYPE_F32: {
        float *out_ptr = reinterpret_cast<float *>(out);
        const float *in_ptr = reinterpret_cast<const float *>(in);
        const float *weight_ptr = reinterpret_cast<const float *>(weight);
        const float *bias_ptr = reinterpret_cast<const float *>(bias);
        linear_<float>(out_ptr, in_ptr, weight_ptr, bias_ptr, M, K, N);
        break;
    }

    case LLAISYS_DTYPE_F16: {
        llaisys::fp16_t *out_ptr = reinterpret_cast<llaisys::fp16_t *>(out);
        const llaisys::fp16_t *in_ptr = reinterpret_cast<const llaisys::fp16_t *>(in);
        const llaisys::fp16_t *weight_ptr = reinterpret_cast<const llaisys::fp16_t *>(weight);
        const llaisys::fp16_t *bias_ptr = reinterpret_cast<const llaisys::fp16_t *>(bias);
        linear_<llaisys::fp16_t>(out_ptr, in_ptr, weight_ptr, bias_ptr, M, K, N);
        break;
    }

    case LLAISYS_DTYPE_BF16: {
        llaisys::bf16_t *out_ptr = reinterpret_cast<llaisys::bf16_t *>(out);
        const llaisys::bf16_t *in_ptr = reinterpret_cast<const llaisys::bf16_t *>(in);
        const llaisys::bf16_t *weight_ptr = reinterpret_cast<const llaisys::bf16_t *>(weight);
        const llaisys::bf16_t *bias_ptr = reinterpret_cast<const llaisys::bf16_t *>(bias);
        linear_<llaisys::bf16_t>(out_ptr, in_ptr, weight_ptr, bias_ptr, M, K, N);
        break;
    }

    default:
        EXCEPTION_UNSUPPORTED_DATATYPE(dtype);
    }
}

} // namespace llaisys::ops::cpu
