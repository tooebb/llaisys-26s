#pragma once
#include "../../../tensor/tensor.hpp"

namespace llaisys::ops::nvidia {
void linear(std::byte *out, const std::byte *in, const std::byte *weight,
            const std::byte *bias, llaisysDataType_t dtype,
            size_t M, size_t K, size_t N);
}
