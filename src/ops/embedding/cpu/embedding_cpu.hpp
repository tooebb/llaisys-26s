#pragma once

#include "llaisys.h"

#include <cstddef>

namespace llaisys::ops::cpu {

/*
 * CPU 版 embedding（查表）
 * 参数：
 *   out    - 输出，2D 张量 [seq_len, dim]
 *   index  - 索引，1D 张量 [seq_len]，int64
 *   weight - 权重表，2D 张量 [vocab_size, dim]
 *   seq_len   - index 的元素个数
 *   dim       - 每个词向量的长度
 *   vocab_size - 词汇表大小
 */
void embedding(std::byte *out, const std::byte *index, const std::byte *weight,
               llaisysDataType_t dtype, size_t seq_len, size_t dim);

} // namespace llaisys::ops::cpu
