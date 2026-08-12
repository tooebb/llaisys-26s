#include "self_attention_nvidia.cuh"

namespace llaisys::ops::nvidia {

constexpr int MAX_THREADS = 256;

// 每个 block 处理一个 (query_head, query_position) 对
template <typename T>
__global__ void self_attention_kernel(
    T *attn_val,
    const T *q, const T *k, const T *v,
    size_t seq_len, size_t n_heads, size_t n_kv_heads,
    size_t d, size_t dv, size_t total_len, float scale) {

    size_t h = blockIdx.y;
    size_t q_pos = blockIdx.x;
    if (h >= n_heads || q_pos >= seq_len) return;

    size_t kv_ratio = n_heads / n_kv_heads;
    size_t kv_h = h / kv_ratio;
    // 因果 mask: 能看到的 KV 位置数
    size_t max_kv = total_len - seq_len + q_pos + 1;

    // 共享内存: Q 行 + 各线程局部 max/sum 归并用
    extern __shared__ float s_buf[];
    float *s_q   = s_buf;           // [d]
    float *s_max = s_buf + d;       // [blockDim.x]
    float *s_sum = s_buf + d + MAX_THREADS; // [blockDim.x]
    float *s_out = s_buf + d;       // 复用: 输出累加 [dv]

    // 阶段 0: 加载 Q 到共享内存
    for (size_t j = threadIdx.x; j < d; j += blockDim.x) {
        s_q[j] = float(q[q_pos * n_heads * d + h * d + j]);
    }
    __syncthreads();

    // 阶段 1: 在线 softmax — 每个线程扫描一部分 KV
    float local_max = -1e30f;
    float local_sum = 0.0f;

    for (size_t j = threadIdx.x; j < max_kv; j += blockDim.x) {
        float dot = 0.0f;
        for (size_t di = 0; di < d; di++) {
            dot += s_q[di] *
                   float(k[j * n_kv_heads * d + kv_h * d + di]);
        }
        dot *= scale;

        float new_max = fmaxf(local_max, dot);
        local_sum = local_sum * expf(local_max - new_max) + expf(dot - new_max);
        local_max = new_max;
    }

    // 归并所有线程的 (max, sum) → 全局 (max, sum)
    s_max[threadIdx.x] = local_max;
    s_sum[threadIdx.x] = local_sum;
    __syncthreads();

    for (size_t stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride) {
            float other_max = s_max[threadIdx.x + stride];
            float other_sum = s_sum[threadIdx.x + stride];
            float my_max = s_max[threadIdx.x];
            float my_sum = s_sum[threadIdx.x];

            float new_max = fmaxf(my_max, other_max);
            s_sum[threadIdx.x] =
                my_sum * expf(my_max - new_max) +
                other_sum * expf(other_max - new_max);
            s_max[threadIdx.x] = new_max;
        }
        __syncthreads();
    }

    float global_max = s_max[0];
    float global_sum = s_sum[0];

    // 阶段 2: 每个线程负责一个 v_dim 的加权累加
    size_t v_dim = threadIdx.x;
    if (v_dim >= dv) return;

    float val = 0.0f;
    for (size_t j = 0; j < max_kv; j++) {
        float dot = 0.0f;
        for (size_t di = 0; di < d; di++) {
            dot += s_q[di] *
                   float(k[j * n_kv_heads * d + kv_h * d + di]);
        }
        float weight = expf(dot * scale - global_max) / global_sum;
        val += weight *
               float(v[j * n_kv_heads * dv + kv_h * dv + v_dim]);
    }

    attn_val[q_pos * n_heads * dv + h * dv + v_dim] = T(val);
}

void self_attention(std::byte *attn_val, const std::byte *q,
                    const std::byte *k, const std::byte *v,
                    llaisysDataType_t dtype, size_t seq_len,
                    size_t n_heads, size_t n_kv_heads,
                    size_t d, size_t dv, size_t total_len,
                    float scale) {
    dim3 grid((unsigned)seq_len, (unsigned)n_heads);
    // 共享内存: d + 2*MAX_THREADS 个 float
    size_t smem = (d + 2 * MAX_THREADS) * sizeof(float);

    switch (dtype) {
    case LLAISYS_DTYPE_F32:
        self_attention_kernel<float><<<grid, MAX_THREADS, smem>>>(
            (float *)attn_val, (const float *)q, (const float *)k,
            (const float *)v, seq_len, n_heads, n_kv_heads, d, dv,
            total_len, scale);
        break;
    case LLAISYS_DTYPE_F16:
        self_attention_kernel<__half><<<grid, MAX_THREADS, smem>>>(
            (__half *)attn_val, (const __half *)q, (const __half *)k,
            (const __half *)v, seq_len, n_heads, n_kv_heads, d, dv,
            total_len, scale);
        break;
    case LLAISYS_DTYPE_BF16:
        self_attention_kernel<__nv_bfloat16><<<grid, MAX_THREADS, smem>>>(
            (__nv_bfloat16 *)attn_val, (const __nv_bfloat16 *)q,
            (const __nv_bfloat16 *)k, (const __nv_bfloat16 *)v,
            seq_len, n_heads, n_kv_heads, d, dv, total_len, scale);
        break;
    default:
        break;
    }
}

} // namespace llaisys::ops::nvidia
