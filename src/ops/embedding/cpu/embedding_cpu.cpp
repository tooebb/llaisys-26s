#include "embedding_cpu.hpp"

#include "../../../utils.hpp"

#include <cstdint>
#include <cstring>

namespace llaisys::ops::cpu {

/* ====== 模板函数：处理一种具体类型 ====== */
template <typename T>
static void embedding_(T *out, const int64_t *index, const T *weight,
                       size_t seq_len, size_t dim) {
    for (size_t i = 0; i < seq_len; i++) {
        int64_t row = index[i];          // 要查哪一行
        /* 把 weight 的第 row 行拷贝到 out 的第 i 行 */
        for (size_t j = 0; j < dim; j++) {
            out[i * dim + j] = weight[row * dim + j];
        }
    }
}

/* ====== 分派函数：根据运行时 dtype 调对应的模板 ====== */
void embedding(std::byte *out, const std::byte *index, const std::byte *weight,
               llaisysDataType_t dtype, size_t seq_len, size_t dim) {

    switch (dtype) {

    case LLAISYS_DTYPE_F32:
        embedding_<float>(
            reinterpret_cast<float *>(out),
            reinterpret_cast<const int64_t *>(index),
            reinterpret_cast<const float *>(weight),
            seq_len, dim);
        break;

    case LLAISYS_DTYPE_F16:
        embedding_<llaisys::fp16_t>(
            reinterpret_cast<llaisys::fp16_t *>(out),
            reinterpret_cast<const int64_t *>(index),
            reinterpret_cast<const llaisys::fp16_t *>(weight),
            seq_len, dim);
        break;

    case LLAISYS_DTYPE_BF16:
        embedding_<llaisys::bf16_t>(
            reinterpret_cast<llaisys::bf16_t *>(out),
            reinterpret_cast<const int64_t *>(index),
            reinterpret_cast<const llaisys::bf16_t *>(weight),
            seq_len, dim);
        break;

    default:
        EXCEPTION_UNSUPPORTED_DATATYPE(dtype);
    }
}

} // namespace llaisys::ops::cpu
