#include "swiglu_cpu.hpp"

#include "../../../utils.hpp"

#include <cmath>

namespace llaisys::ops::cpu {

template <typename T>
static void swiglu_(T *out, const T *gate, const T *up, size_t numel) {
    for (size_t i = 0; i < numel; i++) {
        if constexpr (std::is_same_v<T, llaisys::bf16_t>
                      || std::is_same_v<T, llaisys::fp16_t>) {
            float g = llaisys::utils::cast<float>(gate[i]);
            float u = llaisys::utils::cast<float>(up[i]);
            float act = g / (1.0f + std::exp(-g));
            out[i] = llaisys::utils::cast<T>(u * act);
        } else {
            float act = gate[i] / (1.0f + std::exp(-gate[i]));
            out[i] = up[i] * act;
        }
    }
}

void swiglu(std::byte *out, const std::byte *gate, const std::byte *up,
            llaisysDataType_t dtype, size_t numel) {

    switch (dtype) {

    case LLAISYS_DTYPE_F32: {
        float *out_ptr = reinterpret_cast<float *>(out);
        const float *gate_ptr = reinterpret_cast<const float *>(gate);
        const float *up_ptr = reinterpret_cast<const float *>(up);
        swiglu_<float>(out_ptr, gate_ptr, up_ptr, numel);
        break;
    }

    case LLAISYS_DTYPE_F16: {
        llaisys::fp16_t *out_ptr = reinterpret_cast<llaisys::fp16_t *>(out);
        const llaisys::fp16_t *gate_ptr = reinterpret_cast<const llaisys::fp16_t *>(gate);
        const llaisys::fp16_t *up_ptr = reinterpret_cast<const llaisys::fp16_t *>(up);
        swiglu_<llaisys::fp16_t>(out_ptr, gate_ptr, up_ptr, numel);
        break;
    }

    case LLAISYS_DTYPE_BF16: {
        llaisys::bf16_t *out_ptr = reinterpret_cast<llaisys::bf16_t *>(out);
        const llaisys::bf16_t *gate_ptr = reinterpret_cast<const llaisys::bf16_t *>(gate);
        const llaisys::bf16_t *up_ptr = reinterpret_cast<const llaisys::bf16_t *>(up);
        swiglu_<llaisys::bf16_t>(out_ptr, gate_ptr, up_ptr, numel);
        break;
    }

    default:
        EXCEPTION_UNSUPPORTED_DATATYPE(dtype);
    }
}

} // namespace llaisys::ops::cpu
