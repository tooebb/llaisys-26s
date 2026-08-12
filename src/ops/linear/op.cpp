#include "op.hpp"

#include "../../core/llaisys_core.hpp"
#include "../../utils.hpp"

#include "cpu/linear_cpu.hpp"

#ifdef ENABLE_NVIDIA_API
#include "nvidia/linear_nvidia.cuh"
#endif
#ifdef ENABLE_METAX_API
#include "metax/linear_metax.cuh"
#endif

namespace llaisys::ops {

void linear(tensor_t out, tensor_t in, tensor_t weight, tensor_t bias) {
    CHECK_SAME_DEVICE(out, in, weight);
    CHECK_ARGUMENT(out->ndim() == 2 && in->ndim() == 2 && weight->ndim() == 2,
                   "linear: out, in, weight must be 2D.");
    CHECK_SAME_DTYPE(out->dtype(), in->dtype(), weight->dtype());

    size_t M = in->shape()[0];   // 输入行数
    size_t K = in->shape()[1];   // 输入列数（= 累加维度）
    size_t N = weight->shape()[0]; // 输出列数（weight 形状为 N×K）

    CHECK_ARGUMENT(out->shape()[0] == M && out->shape()[1] == N,
                   "linear: out shape mismatch.");
    CHECK_ARGUMENT(weight->shape()[1] == K,
                   "linear: weight.shape[1] must match in.shape[1].");

    if (bias) {
        CHECK_SAME_DEVICE(out, bias);
        CHECK_SAME_DTYPE(out->dtype(), bias->dtype());
        CHECK_ARGUMENT(bias->ndim() == 1 && bias->shape()[0] == N,
                       "linear: bias must be 1D with size N.");
    }

    /* CPU 实现 */
    if (out->deviceType() == LLAISYS_DEVICE_CPU) {
        const std::byte *bias_data = bias ? bias->data() : nullptr;
        return cpu::linear(out->data(), in->data(), weight->data(),
                           bias_data, out->dtype(), M, K, N);
    }

    /* 其他设备 */
    llaisys::core::context().setDevice(out->deviceType(), out->deviceId());

    switch (out->deviceType()) {

    case LLAISYS_DEVICE_CPU: {
        const std::byte *bias_data = bias ? bias->data() : nullptr;
        return cpu::linear(out->data(), in->data(), weight->data(),
                           bias_data, out->dtype(), M, K, N);
    }

#ifdef ENABLE_NVIDIA_API
    case LLAISYS_DEVICE_NVIDIA: {
        const std::byte *bias_data = bias ? bias->data() : nullptr;
        return nvidia::linear(out->data(), in->data(), weight->data(),
                              bias_data, out->dtype(), M, K, N);
    }
#endif
#ifdef ENABLE_METAX_API
    case LLAISYS_DEVICE_METAX: {
        const std::byte *bias_data = bias ? bias->data() : nullptr;
        return metax::linear(out->data(), in->data(), weight->data(),
                             bias_data, out->dtype(), M, K, N);
    }
#endif

    default:
        EXCEPTION_UNSUPPORTED_DEVICE;
    }
}

} // namespace llaisys::ops
