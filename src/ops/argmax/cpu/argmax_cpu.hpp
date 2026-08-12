#pragma once

#include "llaisys.h"

#include <cstddef>

namespace llaisys::ops::cpu {

/*
 * 参数：
 * max_idx  - 输出，存最大值的索引（int64）
 * max_val  - 输出，存最大值的数值
 * vals     - 输入，要找最大值的数据
 * dtype    - 数据类型（F32 / F16 / BF16）
 * numel    - vals 里有多少个元素
 */
void argmax(std::byte *max_idx, std::byte *max_val, const std::byte *vals,
            llaisysDataType_t dtype, size_t numel);

} // namespace llaisys::ops::cpu
