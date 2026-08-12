#include "op.hpp"

#include "../../core/llaisys_core.hpp"
#include "../../utils.hpp"

#include "cpu/argmax_cpu.hpp"

#ifdef ENABLE_NVIDIA_API
#include "nvidia/argmax_nvidia.cuh"
#endif
#ifdef ENABLE_METAX_API
#include "metax/argmax_metax.cuh"
#endif

namespace llaisys::ops {

void argmax(tensor_t max_idx, tensor_t max_val, tensor_t vals) {
    /* 检查三个张量在不在同一个设备上 */
    CHECK_SAME_DEVICE(max_idx, max_val, vals);

    /* 检查形状和类型 */
    CHECK_ARGUMENT(vals->ndim() == 1,
                   "argmax: vals must be a 1D tensor.");

    CHECK_ARGUMENT(max_val->numel() == 1,
                   "argmax: max_val must be a single-element tensor.");
    CHECK_ARGUMENT(max_idx->numel() == 1,
                   "argmax: max_idx must be a single-element tensor.");

    CHECK_ARGUMENT(max_idx->dtype() == LLAISYS_DTYPE_I64,
                   "argmax: max_idx must be int64 type.");

    CHECK_ARGUMENT(max_val->dtype() == vals->dtype(),
                   "argmax: max_val and vals must have the same dtype.");

    /* 如果是 CPU，直接调 CPU 实现 */
    if (max_val->deviceType() == LLAISYS_DEVICE_CPU) {
        return cpu::argmax(
            max_idx->data(),
            max_val->data(),
            vals->data(),
            vals->dtype(),
            vals->numel());
    }

    /* 如果不是 CPU，则尝试其他设备 */
    llaisys::core::context().setDevice(max_val->deviceType(), max_val->deviceId());

    switch (max_val->deviceType()) {

    case LLAISYS_DEVICE_CPU:
        return cpu::argmax(max_idx->data(), max_val->data(), vals->data(),
                           vals->dtype(), vals->numel());

#ifdef ENABLE_NVIDIA_API
    case LLAISYS_DEVICE_NVIDIA:
        return nvidia::argmax(max_idx->data(), max_val->data(), vals->data(),
                              vals->dtype(), vals->numel());
#endif
#ifdef ENABLE_METAX_API
    case LLAISYS_DEVICE_METAX:
        return metax::argmax(max_idx->data(), max_val->data(), vals->data(),
                             vals->dtype(), vals->numel());
#endif

    default:
        EXCEPTION_UNSUPPORTED_DEVICE;
    }
}

} // namespace llaisys::ops
