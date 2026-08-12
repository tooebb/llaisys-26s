#include "../runtime_api.hpp"

#include <mc_runtime.h>

namespace llaisys::device::metax {

namespace runtime_api {

static mcMemcpyKind_t to_mc_kind(llaisysMemcpyKind_t kind) {
    switch (kind) {
    case LLAISYS_MEMCPY_H2H: return mcMemcpyHostToHost;
    case LLAISYS_MEMCPY_H2D: return mcMemcpyHostToDevice;
    case LLAISYS_MEMCPY_D2H: return mcMemcpyDeviceToHost;
    case LLAISYS_MEMCPY_D2D: return mcMemcpyDeviceToDevice;
    default:                  return mcMemcpyDefault;
    }
}

int getDeviceCount() {
    int count = 0;
    mcGetDeviceCount(&count);
    return count;
}

void setDevice(int device_id) {
    mcSetDevice(device_id);
}

void deviceSynchronize() {
    mcDeviceSynchronize();
}

llaisysStream_t createStream() {
    mcStream_t stream = nullptr;
    mcStreamCreate(&stream);
    return (llaisysStream_t)stream;
}

void destroyStream(llaisysStream_t stream) {
    mcStreamDestroy((mcStream_t)stream);
}

void streamSynchronize(llaisysStream_t stream) {
    mcStreamSynchronize((mcStream_t)stream);
}

void *mallocDevice(size_t size) {
    void *ptr = nullptr;
    mcMalloc(&ptr, size);
    return ptr;
}

void freeDevice(void *ptr) {
    mcFree(ptr);
}

void *mallocHost(size_t size) {
    void *ptr = nullptr;
    mcMallocHost(&ptr, size);
    return ptr;
}

void freeHost(void *ptr) {
    mcFreeHost(ptr);
}

void memcpySync(void *dst, const void *src, size_t size, llaisysMemcpyKind_t kind) {
    mcMemcpy(dst, src, size, to_mc_kind(kind));
}

void memcpyAsync(void *dst, const void *src, size_t size,
                 llaisysMemcpyKind_t kind, llaisysStream_t stream) {
    mcMemcpyAsync(dst, src, size, to_mc_kind(kind), (mcStream_t)stream);
}

static const LlaisysRuntimeAPI RUNTIME_API = {
    &getDeviceCount,
    &setDevice,
    &deviceSynchronize,
    &createStream,
    &destroyStream,
    &streamSynchronize,
    &mallocDevice,
    &freeDevice,
    &mallocHost,
    &freeHost,
    &memcpySync,
    &memcpyAsync};

} // namespace runtime_api

const LlaisysRuntimeAPI *getRuntimeAPI() {
    return &runtime_api::RUNTIME_API;
}
} // namespace llaisys::device::metax
