#include "linear_metax.cuh"

#include <mcblas.h>

namespace llaisys::ops::metax {

constexpr int TILE = 16;
constexpr int BLOCK_SIZE = 256;

template <typename T>
__global__ void add_bias_kernel(T *out, const T *bias, size_t M, size_t N) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    size_t total = M * N;
    if (idx >= total) return;
    size_t col = idx % N;
    out[idx] = out[idx] + bias[col];
}

// BF16 tiled GEMM
__global__ void bf16_gemm_kernel(__nv_bfloat16 *out,
                                  const __nv_bfloat16 *in,
                                  const __nv_bfloat16 *weight,
                                  size_t M, size_t K, size_t N) {
    int row = blockIdx.y * TILE + threadIdx.y;
    int col = blockIdx.x * TILE + threadIdx.x;
    if (row >= (int)M || col >= (int)N) return;

    float sum = 0.0f;
    for (size_t k = 0; k < K; k++) {
        sum += float(in[row * K + k]) * float(weight[k * N + col]);
    }
    out[row * N + col] = __nv_bfloat16(sum);
}

static void linear_bf16_impl(__nv_bfloat16 *out, const __nv_bfloat16 *in,
                              const __nv_bfloat16 *weight,
                              const __nv_bfloat16 *bias,
                              size_t M, size_t K, size_t N) {
    dim3 block(TILE, TILE);
    dim3 grid((unsigned)((N + TILE - 1) / TILE),
              (unsigned)((M + TILE - 1) / TILE));
    bf16_gemm_kernel<<<grid, block>>>(out, in, weight, M, K, N);

    if (bias) {
        size_t total = M * N;
        size_t g = (total + BLOCK_SIZE - 1) / BLOCK_SIZE;
        add_bias_kernel<__nv_bfloat16><<<g, BLOCK_SIZE>>>(out, bias, M, N);
    }
}

// F32: mcBLAS GEMM
static void linear_f32_impl(float *out, const float *in, const float *weight,
                             const float *bias, size_t M, size_t K, size_t N) {
    mcblasHandle_t handle;
    mcblasCreate(&handle);

    float alpha = 1.0f, beta = 0.0f;
    mcblasSgemm(handle,
                MCBLAS_OP_N, MCBLAS_OP_N,
                (int)N, (int)M, (int)K,
                &alpha,
                weight, (int)N,
                in, (int)K,
                &beta,
                out, (int)N);

    if (bias) {
        size_t total = M * N;
        size_t g = (total + BLOCK_SIZE - 1) / BLOCK_SIZE;
        add_bias_kernel<float><<<g, BLOCK_SIZE>>>(out, bias, M, N);
    }

    mcblasDestroy(handle);
}

// F16: mcBLAS GEMM
static void linear_f16_impl(__half *out, const __half *in, const __half *weight,
                             const __half *bias, size_t M, size_t K, size_t N) {
    mcblasHandle_t handle;
    mcblasCreate(&handle);

    __half alpha = __float2half(1.0f);
    __half beta  = __float2half(0.0f);
    mcblasHgemm(handle,
                MCBLAS_OP_N, MCBLAS_OP_N,
                (int)N, (int)M, (int)K,
                &alpha,
                weight, (int)N,
                in, (int)K,
                &beta,
                out, (int)N);

    if (bias) {
        size_t total = M * N;
        size_t g = (total + BLOCK_SIZE - 1) / BLOCK_SIZE;
        add_bias_kernel<__half><<<g, BLOCK_SIZE>>>(out, bias, M, N);
    }

    mcblasDestroy(handle);
}

void linear(std::byte *out, const std::byte *in, const std::byte *weight,
            const std::byte *bias, llaisysDataType_t dtype,
            size_t M, size_t K, size_t N) {
    switch (dtype) {
    case LLAISYS_DTYPE_F32:
        linear_f32_impl((float *)out, (const float *)in, (const float *)weight,
                        (const float *)bias, M, K, N);
        break;
    case LLAISYS_DTYPE_F16:
        linear_f16_impl((__half *)out, (const __half *)in, (const __half *)weight,
                        (const __half *)bias, M, K, N);
        break;
    case LLAISYS_DTYPE_BF16:
        linear_bf16_impl((__nv_bfloat16 *)out, (const __nv_bfloat16 *)in,
                         (const __nv_bfloat16 *)weight,
                         (const __nv_bfloat16 *)bias, M, K, N);
        break;
    default:
        break;
    }
}

} // namespace llaisys::ops::metax
