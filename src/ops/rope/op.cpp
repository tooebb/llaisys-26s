#include "op.hpp"

#include "../../core/llaisys_core.hpp"
#include "../../utils.hpp"

#include "cpu/rope_cpu.hpp"

namespace llaisys::ops {

void rope(tensor_t out, tensor_t in, tensor_t pos_ids, float theta) {
    CHECK_SAME_DEVICE(out, in, pos_ids);
    CHECK_SAME_SHAPE(out->shape(), in->shape());
    CHECK_SAME_DTYPE(out->dtype(), in->dtype());

    CHECK_ARGUMENT(out->ndim() == 3,
                   "rope: out/in must be 3D [seq_len, n_heads, dim].");
    CHECK_ARGUMENT(pos_ids->ndim() == 1 && pos_ids->dtype() == LLAISYS_DTYPE_I64,
                   "rope: pos_ids must be 1D int64.");
    CHECK_ARGUMENT(pos_ids->shape()[0] == out->shape()[0],
                   "rope: pos_ids size must match seq_len.");

    size_t seq_len = out->shape()[0];
    size_t n_heads = out->shape()[1];
    size_t dim = out->shape()[2];

    if (out->deviceType() == LLAISYS_DEVICE_CPU) {
        return cpu::rope(out->data(), in->data(), pos_ids->data(),
                         out->dtype(), seq_len, n_heads, dim, theta);
    }

    llaisys::core::context().setDevice(out->deviceType(), out->deviceId());

    switch (out->deviceType()) {

    case LLAISYS_DEVICE_CPU:
        return cpu::rope(out->data(), in->data(), pos_ids->data(),
                         out->dtype(), seq_len, n_heads, dim, theta);

#ifdef ENABLE_NVIDIA_API
    case LLAISYS_DEVICE_NVIDIA:
        TO_BE_IMPLEMENTED();
        return;
#endif

    default:
        EXCEPTION_UNSUPPORTED_DEVICE;
    }
}

} // namespace llaisys::ops
