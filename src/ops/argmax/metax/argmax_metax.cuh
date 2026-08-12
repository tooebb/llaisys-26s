#pragma once
#include "../../../tensor/tensor.hpp"

namespace llaisys::ops::metax {
void argmax(std::byte *max_idx, std::byte *max_val, const std::byte *vals,
            llaisysDataType_t dtype, size_t numel);
}
