#include "op.hpp"

#include "../../core/llaisys_core.hpp"
#include "../../utils.hpp"

#include "cpu/embedding_cpu.hpp"

#ifdef ENABLE_NVIDIA_API
#include "nvidia/embedding_nvidia.cuh"
#endif
#ifdef ENABLE_METAX_API
#include "metax/embedding_metax.cuh"
#endif

namespace llaisys::ops {

void embedding(tensor_t out, tensor_t index, tensor_t weight) {
    /* 检查设备是否一致 */
    CHECK_SAME_DEVICE(out, index, weight);

    /* 检查形状 */
    CHECK_ARGUMENT(index->ndim() == 1,
                   "embedding: index must be a 1D tensor.");
    CHECK_ARGUMENT(weight->ndim() == 2,
                   "embedding: weight must be a 2D tensor.");
    CHECK_ARGUMENT(out->ndim() == 2,
                   "embedding: out must be a 2D tensor.");

    size_t seq_len = index->numel();           // 查多少次
    size_t dim = weight->shape()[1];            // 每个词向量的长度

    CHECK_ARGUMENT(out->shape()[0] == seq_len,
                   "embedding: out.shape[0] must match index size.");
    CHECK_ARGUMENT(out->shape()[1] == dim,
                   "embedding: out.shape[1] must match weight.shape[1].");

    /* 检查类型 */
    CHECK_ARGUMENT(index->dtype() == LLAISYS_DTYPE_I64,
                   "embedding: index must be int64 type.");
    CHECK_ARGUMENT(out->dtype() == weight->dtype(),
                   "embedding: out and weight must have the same dtype.");

    /* CPU 实现 */
    if (out->deviceType() == LLAISYS_DEVICE_CPU) {
        return cpu::embedding(
            out->data(),
            index->data(),
            weight->data(),
            out->dtype(),
            seq_len,
            dim);
    }

    /* 其他设备 */
    llaisys::core::context().setDevice(out->deviceType(), out->deviceId());

    switch (out->deviceType()) {

    case LLAISYS_DEVICE_CPU:
        return cpu::embedding(out->data(), index->data(), weight->data(),
                              out->dtype(), seq_len, dim);

#ifdef ENABLE_NVIDIA_API
    case LLAISYS_DEVICE_NVIDIA:
        return nvidia::embedding(out->data(), index->data(), weight->data(),
                                 out->dtype(), seq_len, dim);
#endif
#ifdef ENABLE_METAX_API
    case LLAISYS_DEVICE_METAX:
        return metax::embedding(out->data(), index->data(), weight->data(),
                                out->dtype(), seq_len, dim);
#endif

    default:
        EXCEPTION_UNSUPPORTED_DEVICE;
    }
}

} // namespace llaisys::ops
