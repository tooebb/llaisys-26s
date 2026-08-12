#include "argmax_cpu.hpp"

#include "../../../utils.hpp"

#include <cstdint>

namespace llaisys::ops::cpu {
/*
 * T 是数据类型，可能是 float / fp16_t / bf16_t
 * max_idx 永远是 int64_t
 */
template <typename T>
static void argmax_(int64_t *max_idx, T *max_val, const T *vals, size_t numel) {
    /* 假设第 0 个元素最大，往后逐个比较 */
    T best_val = vals[0];
    int64_t best_idx = 0;

    for (size_t i = 1; i < numel; i++) {
        /*
         * F16 / BF16 不能用 > 直接比较，要先转成 float
         * 普通 float 直接比就行
         */
        if constexpr (std::is_same_v<T, llaisys::bf16_t>
                      || std::is_same_v<T, llaisys::fp16_t>) {
            /* 半精度：转成 float 再比较 */
            if (llaisys::utils::cast<float>(vals[i])
                > llaisys::utils::cast<float>(best_val)) {
                best_val = vals[i];   // 存原始值（保持原类型）
                best_idx = (int64_t)i;
            }
        } else {
            /* 普通类型：直接比较 */
            if (vals[i] > best_val) {
                best_val = vals[i];
                best_idx = (int64_t)i;
            }
        }
    }

    *max_idx = best_idx;
    *max_val = best_val;
}

/*分派函数：根据运行时 dtype 调对应的模板*/
void argmax(std::byte *max_idx, std::byte *max_val, const std::byte *vals,
            llaisysDataType_t dtype, size_t numel) {

    switch (dtype) {

    case LLAISYS_DTYPE_F32:
        argmax_<float>(
            reinterpret_cast<int64_t *>(max_idx),   // max_idx 永远用 int64 解读
            reinterpret_cast<float *>(max_val),     // max_val 用 float 解读
            reinterpret_cast<const float *>(vals),  // vals 用 float 解读
            numel);
        break;

    case LLAISYS_DTYPE_F16:
        argmax_<llaisys::fp16_t>(
            reinterpret_cast<int64_t *>(max_idx),
            reinterpret_cast<llaisys::fp16_t *>(max_val),
            reinterpret_cast<const llaisys::fp16_t *>(vals),
            numel);
        break;

    case LLAISYS_DTYPE_BF16:
        argmax_<llaisys::bf16_t>(
            reinterpret_cast<int64_t *>(max_idx),
            reinterpret_cast<llaisys::bf16_t *>(max_val),
            reinterpret_cast<const llaisys::bf16_t *>(vals),
            numel);
        break;

    default:
        EXCEPTION_UNSUPPORTED_DATATYPE(dtype);  // 不支持的类型就报错
    }
}

} // namespace llaisys::ops::cpu
