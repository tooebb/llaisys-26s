#include "nvidia_resource.cuh"

#include <cuda_runtime.h>

namespace llaisys::device::nvidia {

Resource::Resource(int device_id)
    : llaisys::device::DeviceResource(LLAISYS_DEVICE_NVIDIA, device_id),
      cublas_handle(nullptr) {
    cublasCreate(&cublas_handle);
}

Resource::~Resource() {
    if (cublas_handle) {
        cublasDestroy(cublas_handle);
    }
}

} // namespace llaisys::device::nvidia
