#pragma once

#include "../../core/llaisys_core.hpp"
#include "../../tensor/tensor.hpp"

#include "llaisys.h"
#include "llaisys/models/qwen2.h"

#include <cstdint>
#include <vector>

namespace llaisys::models {

// 单层权重
struct Qwen2LayerWeights {
    tensor_t attn_norm_w;
    tensor_t attn_q_w, attn_q_b;
    tensor_t attn_k_w, attn_k_b;
    tensor_t attn_v_w, attn_v_b;
    tensor_t attn_o_w;
    tensor_t mlp_norm_w;
    tensor_t mlp_gate_w, mlp_up_w, mlp_down_w;
};

// Qwen2 模型
struct Qwen2Model {
    LlaisysQwen2Meta meta;
    tensor_t in_embed;
    tensor_t out_embed;
    tensor_t out_norm_w;
    std::vector<Qwen2LayerWeights> layers;

    // KV cache: 每层一份
    std::vector<tensor_t> k_caches;
    std::vector<tensor_t> v_caches;
    size_t cache_len;

    // 临时张量（复用避免反复分配）
    tensor_t q_flat, k_flat, v_flat;
    tensor_t q_temp, k_temp, v_temp, attn_temp;
    tensor_t attn_out, attn_o_out;
    tensor_t normed, normed2;
    tensor_t gate_out, up_out, swiglu_out, mlp_out;
    tensor_t logits;
    tensor_t max_idx, max_val;
    tensor_t pos_ids;

    void init_cache(llaisysDeviceType_t device, int device_id);
    void forward(const int64_t *token_ids, size_t ntoken, int64_t *out_token);
};

} // namespace llaisys::models
