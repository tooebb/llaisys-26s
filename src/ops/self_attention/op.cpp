#include "op.hpp"

#include "../../core/llaisys_core.hpp"
#include "../../utils.hpp"

#include "cpu/self_attention_cpu.hpp"

#ifdef ENABLE_NVIDIA_API
#include "nvidia/self_attention_nvidia.cuh"
#endif
#ifdef ENABLE_METAX_API
#include "metax/self_attention_metax.cuh"
#endif

namespace llaisys::ops {

void self_attention(tensor_t attn_val, tensor_t q, tensor_t k, tensor_t v, float scale) {
    CHECK_SAME_DEVICE(attn_val, q, k, v);
    CHECK_SAME_DTYPE(attn_val->dtype(), q->dtype(), k->dtype(), v->dtype());

    size_t seq_len = q->shape()[0];
    size_t n_heads = q->shape()[1];
    size_t d = q->shape()[2];

    size_t total_len = k->shape()[0];
    size_t n_kv_heads = k->shape()[1];
    size_t dv = v->shape()[2];

    CHECK_ARGUMENT(attn_val->shape()[0] == seq_len
                   && attn_val->shape()[1] == n_heads
                   && attn_val->shape()[2] == dv,
                   "self_attention: attn_val shape mismatch.");

    if (attn_val->deviceType() == LLAISYS_DEVICE_CPU) {
        return cpu::self_attention(
            attn_val->data(), q->data(), k->data(), v->data(),
            attn_val->dtype(), seq_len, n_heads, n_kv_heads,
            d, dv, total_len, scale);
    }

    llaisys::core::context().setDevice(attn_val->deviceType(), attn_val->deviceId());

    switch (attn_val->deviceType()) {

    case LLAISYS_DEVICE_CPU:
        return cpu::self_attention(
            attn_val->data(), q->data(), k->data(), v->data(),
            attn_val->dtype(), seq_len, n_heads, n_kv_heads,
            d, dv, total_len, scale);

#ifdef ENABLE_NVIDIA_API
    case LLAISYS_DEVICE_NVIDIA:
        return nvidia::self_attention(
            attn_val->data(), q->data(), k->data(), v->data(),
            attn_val->dtype(), seq_len, n_heads, n_kv_heads,
            d, dv, total_len, scale);
#endif
#ifdef ENABLE_METAX_API
    case LLAISYS_DEVICE_METAX:
        return metax::self_attention(
            attn_val->data(), q->data(), k->data(), v->data(),
            attn_val->dtype(), seq_len, n_heads, n_kv_heads,
            d, dv, total_len, scale);
#endif

    default:
        EXCEPTION_UNSUPPORTED_DEVICE;
    }
}

} // namespace llaisys::ops
