#pragma once
#include "../../../tensor/tensor.hpp"

namespace llaisys::ops::metax {
void swiglu(std::byte *out, const std::byte *gate, const std::byte *up,
            llaisysDataType_t dtype, size_t numel);
}
