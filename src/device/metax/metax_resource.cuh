#pragma once

#include "../device_resource.hpp"

#include <mcblas.h>

namespace llaisys::device::metax {
class Resource : public llaisys::device::DeviceResource {
public:
    mcblasHandle_t mcblas_handle;

    Resource(int device_id);
    ~Resource();
};
} // namespace llaisys::device::metax
