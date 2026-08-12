#include "llaisys/models/qwen2.h"
#include "llaisys_tensor.hpp"
#include "../models/qwen2/model.hpp"

#include <vector>

/* C++ 模型 + C API 权重句柄的包装 */
struct Qwen2ModelCWrapper {
    llaisys::models::Qwen2Model model;
    LlaisysQwen2Weights weights;

    LlaisysTensor in_embed_wrap;
    LlaisysTensor out_embed_wrap;
    LlaisysTensor out_norm_wrap;

    std::vector<LlaisysTensor> attn_norm_w_wrap;
    std::vector<LlaisysTensor> attn_q_w_wrap;
    std::vector<LlaisysTensor> attn_q_b_wrap;
    std::vector<LlaisysTensor> attn_k_w_wrap;
    std::vector<LlaisysTensor> attn_k_b_wrap;
    std::vector<LlaisysTensor> attn_v_w_wrap;
    std::vector<LlaisysTensor> attn_v_b_wrap;
    std::vector<LlaisysTensor> attn_o_w_wrap;
    std::vector<LlaisysTensor> mlp_norm_w_wrap;
    std::vector<LlaisysTensor> mlp_gate_w_wrap;
    std::vector<LlaisysTensor> mlp_up_w_wrap;
    std::vector<LlaisysTensor> mlp_down_w_wrap;

    std::vector<llaisysTensor_t> attn_norm_w_arr;
    std::vector<llaisysTensor_t> attn_q_w_arr;
    std::vector<llaisysTensor_t> attn_q_b_arr;
    std::vector<llaisysTensor_t> attn_k_w_arr;
    std::vector<llaisysTensor_t> attn_k_b_arr;
    std::vector<llaisysTensor_t> attn_v_w_arr;
    std::vector<llaisysTensor_t> attn_v_b_arr;
    std::vector<llaisysTensor_t> attn_o_w_arr;
    std::vector<llaisysTensor_t> mlp_norm_w_arr;
    std::vector<llaisysTensor_t> mlp_gate_w_arr;
    std::vector<llaisysTensor_t> mlp_up_w_arr;
    std::vector<llaisysTensor_t> mlp_down_w_arr;

    Qwen2ModelCWrapper(size_t nlayer) {
        attn_norm_w_wrap.resize(nlayer);
        attn_q_w_wrap.resize(nlayer);
        attn_q_b_wrap.resize(nlayer);
        attn_k_w_wrap.resize(nlayer);
        attn_k_b_wrap.resize(nlayer);
        attn_v_w_wrap.resize(nlayer);
        attn_v_b_wrap.resize(nlayer);
        attn_o_w_wrap.resize(nlayer);
        mlp_norm_w_wrap.resize(nlayer);
        mlp_gate_w_wrap.resize(nlayer);
        mlp_up_w_wrap.resize(nlayer);
        mlp_down_w_wrap.resize(nlayer);

        attn_norm_w_arr.resize(nlayer);
        attn_q_w_arr.resize(nlayer);
        attn_q_b_arr.resize(nlayer);
        attn_k_w_arr.resize(nlayer);
        attn_k_b_arr.resize(nlayer);
        attn_v_w_arr.resize(nlayer);
        attn_v_b_arr.resize(nlayer);
        attn_o_w_arr.resize(nlayer);
        mlp_norm_w_arr.resize(nlayer);
        mlp_gate_w_arr.resize(nlayer);
        mlp_up_w_arr.resize(nlayer);
        mlp_down_w_arr.resize(nlayer);

        weights.attn_norm_w = attn_norm_w_arr.data();
        weights.attn_q_w    = attn_q_w_arr.data();
        weights.attn_q_b    = attn_q_b_arr.data();
        weights.attn_k_w    = attn_k_w_arr.data();
        weights.attn_k_b    = attn_k_b_arr.data();
        weights.attn_v_w    = attn_v_w_arr.data();
        weights.attn_v_b    = attn_v_b_arr.data();
        weights.attn_o_w    = attn_o_w_arr.data();
        weights.mlp_norm_w  = mlp_norm_w_arr.data();
        weights.mlp_gate_w  = mlp_gate_w_arr.data();
        weights.mlp_up_w    = mlp_up_w_arr.data();
        weights.mlp_down_w  = mlp_down_w_arr.data();
    }
};

__C {

__export struct LlaisysQwen2Model *llaisysQwen2ModelCreate(
    const LlaisysQwen2Meta *meta, llaisysDeviceType_t device,
    int *device_ids, int ndevice)
{
    int device_id = (ndevice > 0) ? device_ids[0] : 0;

    auto *w = new Qwen2ModelCWrapper(meta->nlayer);
    auto &m = w->model;
    m.meta = *meta;

    size_t nh_dh   = meta->nh * meta->dh;
    size_t nkvh_dh = meta->nkvh * meta->dh;

    /* 全局权重 */
    m.in_embed = llaisys::Tensor::create(
        {meta->voc, meta->hs}, meta->dtype, device, device_id);
    w->in_embed_wrap.tensor = m.in_embed;
    w->weights.in_embed = &w->in_embed_wrap;

    m.out_embed = llaisys::Tensor::create(
        {meta->voc, meta->hs}, meta->dtype, device, device_id);
    w->out_embed_wrap.tensor = m.out_embed;
    w->weights.out_embed = &w->out_embed_wrap;

    m.out_norm_w = llaisys::Tensor::create(
        {meta->hs}, meta->dtype, device, device_id);
    w->out_norm_wrap.tensor = m.out_norm_w;
    w->weights.out_norm_w = &w->out_norm_wrap;

    /* 每层权重 */
    m.layers.resize(meta->nlayer);
    for (size_t l = 0; l < meta->nlayer; l++) {
        auto &L = m.layers[l];

        L.attn_norm_w = llaisys::Tensor::create(
            {meta->hs}, meta->dtype, device, device_id);
        w->attn_norm_w_wrap[l].tensor = L.attn_norm_w;
        w->attn_norm_w_arr[l] = &w->attn_norm_w_wrap[l];

        L.attn_q_w = llaisys::Tensor::create(
            {nh_dh, meta->hs}, meta->dtype, device, device_id);
        w->attn_q_w_wrap[l].tensor = L.attn_q_w;
        w->attn_q_w_arr[l] = &w->attn_q_w_wrap[l];

        L.attn_q_b = llaisys::Tensor::create(
            {nh_dh}, meta->dtype, device, device_id);
        w->attn_q_b_wrap[l].tensor = L.attn_q_b;
        w->attn_q_b_arr[l] = &w->attn_q_b_wrap[l];

        L.attn_k_w = llaisys::Tensor::create(
            {nkvh_dh, meta->hs}, meta->dtype, device, device_id);
        w->attn_k_w_wrap[l].tensor = L.attn_k_w;
        w->attn_k_w_arr[l] = &w->attn_k_w_wrap[l];

        L.attn_k_b = llaisys::Tensor::create(
            {nkvh_dh}, meta->dtype, device, device_id);
        w->attn_k_b_wrap[l].tensor = L.attn_k_b;
        w->attn_k_b_arr[l] = &w->attn_k_b_wrap[l];

        L.attn_v_w = llaisys::Tensor::create(
            {nkvh_dh, meta->hs}, meta->dtype, device, device_id);
        w->attn_v_w_wrap[l].tensor = L.attn_v_w;
        w->attn_v_w_arr[l] = &w->attn_v_w_wrap[l];

        L.attn_v_b = llaisys::Tensor::create(
            {nkvh_dh}, meta->dtype, device, device_id);
        w->attn_v_b_wrap[l].tensor = L.attn_v_b;
        w->attn_v_b_arr[l] = &w->attn_v_b_wrap[l];

        L.attn_o_w = llaisys::Tensor::create(
            {meta->hs, nh_dh}, meta->dtype, device, device_id);
        w->attn_o_w_wrap[l].tensor = L.attn_o_w;
        w->attn_o_w_arr[l] = &w->attn_o_w_wrap[l];

        L.mlp_norm_w = llaisys::Tensor::create(
            {meta->hs}, meta->dtype, device, device_id);
        w->mlp_norm_w_wrap[l].tensor = L.mlp_norm_w;
        w->mlp_norm_w_arr[l] = &w->mlp_norm_w_wrap[l];

        L.mlp_gate_w = llaisys::Tensor::create(
            {meta->di, meta->hs}, meta->dtype, device, device_id);
        w->mlp_gate_w_wrap[l].tensor = L.mlp_gate_w;
        w->mlp_gate_w_arr[l] = &w->mlp_gate_w_wrap[l];

        L.mlp_up_w = llaisys::Tensor::create(
            {meta->di, meta->hs}, meta->dtype, device, device_id);
        w->mlp_up_w_wrap[l].tensor = L.mlp_up_w;
        w->mlp_up_w_arr[l] = &w->mlp_up_w_wrap[l];

        L.mlp_down_w = llaisys::Tensor::create(
            {meta->hs, meta->di}, meta->dtype, device, device_id);
        w->mlp_down_w_wrap[l].tensor = L.mlp_down_w;
        w->mlp_down_w_arr[l] = &w->mlp_down_w_wrap[l];
    }

    m.init_cache(device, device_id);

    return reinterpret_cast<struct LlaisysQwen2Model *>(w);
}

__export void llaisysQwen2ModelDestroy(struct LlaisysQwen2Model *model) {
    auto *w = reinterpret_cast<Qwen2ModelCWrapper *>(model);
    delete w;
}

__export struct LlaisysQwen2Weights *llaisysQwen2ModelWeights(
    struct LlaisysQwen2Model *model)
{
    auto *w = reinterpret_cast<Qwen2ModelCWrapper *>(model);
    return &w->weights;
}

__export int64_t llaisysQwen2ModelInfer(
    struct LlaisysQwen2Model *model, int64_t *token_ids, size_t ntoken)
{
    auto *w = reinterpret_cast<Qwen2ModelCWrapper *>(model);
    int64_t out_token = 0;
    w->model.forward(token_ids, ntoken, &out_token);
    return out_token;
}

} // extern "C"
