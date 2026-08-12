#pragma once
#include "../../../tensor/tensor.hpp"

namespace llaisys::ops::metax {
void embedding(std::byte *out, const std::byte *index, const std::byte *weight,
               llaisysDataType_t dtype, size_t seq_len, size_t dim);
}
