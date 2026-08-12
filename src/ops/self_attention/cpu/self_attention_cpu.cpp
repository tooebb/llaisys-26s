#include "self_attention_cpu.hpp"

#include "../../../utils.hpp"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <vector>

namespace llaisys::ops::cpu {

template <typename T>
static void self_attention_(T *attn, const T *q, const T *k, const T *v,
                            size_t seq_len, size_t n_heads, size_t n_kv_heads,
                            size_t d, size_t dv, size_t total_len,
                            float scale) {
    size_t kv_ratio = n_heads / n_kv_heads;
    size_t q_head_stride = n_heads * d;
    size_t k_head_stride = n_kv_heads * d;
    size_t v_head_stride = n_kv_heads * dv;
    size_t attn_head_stride = n_heads * dv;
    size_t start_pos = total_len - seq_len;

    std::vector<float> scores(total_len);
    std::vector<float> tmp_out(dv);

    for (size_t h = 0; h < n_heads; h++) {
        size_t kv_h = h / kv_ratio;

        for (size_t i = 0; i < seq_len; i++) {
            std::fill(tmp_out.begin(), tmp_out.end(), 0.0f);

            /* Step 1: 计算 QK^T，找最大值 */
            float max_score = -INFINITY;
            size_t causal_end = i + start_pos;  // 可以 attend 到 start_pos + i
            for (size_t j = 0; j <= causal_end; j++) {
                float score = 0.0f;
                for (size_t dim_e = 0; dim_e < d; dim_e++) {
                    float q_val = llaisys::utils::cast<float>(
                        q[i * q_head_stride + h * d + dim_e]);
                    float k_val = llaisys::utils::cast<float>(
                        k[j * k_head_stride + kv_h * d + dim_e]);
                    score += q_val * k_val;
                }
                score *= scale;
                scores[j] = score;
                if (score > max_score) {
                    max_score = score;
                }
            }

            /* Step 2: Softmax + 加权求和 */
            float sum_exp = 0.0f;
            for (size_t j = 0; j <= causal_end; j++) {
                float weight = std::exp(scores[j] - max_score);
                sum_exp += weight;
                for (size_t dv_e = 0; dv_e < dv; dv_e++) {
                    float v_val = llaisys::utils::cast<float>(
                        v[j * v_head_stride + kv_h * dv + dv_e]);
                    tmp_out[dv_e] += weight * v_val;
                }
            }

            /* Step 3: 归一化并写回 */
            for (size_t dv_e = 0; dv_e < dv; dv_e++) {
                attn[i * attn_head_stride + h * dv + dv_e] =
                    llaisys::utils::cast<T>(tmp_out[dv_e] / sum_exp);
            }
        }
    }
}

void self_attention(std::byte *attn_val, const std::byte *q,
                    const std::byte *k, const std::byte *v,
                    llaisysDataType_t dtype, size_t seq_len,
                    size_t n_heads, size_t n_kv_heads,
                    size_t d, size_t dv, size_t total_len,
                    float scale) {

    switch (dtype) {

    case LLAISYS_DTYPE_F32: {
        float *a_ptr = reinterpret_cast<float *>(attn_val);
        const float *q_ptr = reinterpret_cast<const float *>(q);
        const float *k_ptr = reinterpret_cast<const float *>(k);
        const float *v_ptr = reinterpret_cast<const float *>(v);
        self_attention_<float>(a_ptr, q_ptr, k_ptr, v_ptr,
                               seq_len, n_heads, n_kv_heads, d, dv,
                               total_len, scale);
        break;
    }

    case LLAISYS_DTYPE_F16: {
        llaisys::fp16_t *a_ptr = reinterpret_cast<llaisys::fp16_t *>(attn_val);
        const llaisys::fp16_t *q_ptr = reinterpret_cast<const llaisys::fp16_t *>(q);
        const llaisys::fp16_t *k_ptr = reinterpret_cast<const llaisys::fp16_t *>(k);
        const llaisys::fp16_t *v_ptr = reinterpret_cast<const llaisys::fp16_t *>(v);
        self_attention_<llaisys::fp16_t>(a_ptr, q_ptr, k_ptr, v_ptr,
                                         seq_len, n_heads, n_kv_heads, d, dv,
                                         total_len, scale);
        break;
    }

    case LLAISYS_DTYPE_BF16: {
        llaisys::bf16_t *a_ptr = reinterpret_cast<llaisys::bf16_t *>(attn_val);
        const llaisys::bf16_t *q_ptr = reinterpret_cast<const llaisys::bf16_t *>(q);
        const llaisys::bf16_t *k_ptr = reinterpret_cast<const llaisys::bf16_t *>(k);
        const llaisys::bf16_t *v_ptr = reinterpret_cast<const llaisys::bf16_t *>(v);
        self_attention_<llaisys::bf16_t>(a_ptr, q_ptr, k_ptr, v_ptr,
                                         seq_len, n_heads, n_kv_heads, d, dv,
                                         total_len, scale);
        break;
    }

    default:
        EXCEPTION_UNSUPPORTED_DATATYPE(dtype);
    }
}

} // namespace llaisys::ops::cpu
