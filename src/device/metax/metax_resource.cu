#include "metax_resource.cuh"

#include <mcr/mc_runtime.h>

namespace llaisys::device::metax {

Resource::Resource(int device_id)
    : llaisys::device::DeviceResource(LLAISYS_DEVICE_METAX, device_id),
      mcblas_handle(nullptr) {
    mcblasCreate(&mcblas_handle);
}

Resource::~Resource() {
    if (mcblas_handle) {
        mcblasDestroy(mcblas_handle);
    }
}

} // namespace llaisys::device::metax
