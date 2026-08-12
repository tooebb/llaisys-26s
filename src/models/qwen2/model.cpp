#include "model.hpp"

#include "../../ops/add/op.hpp"
#include "../../ops/argmax/op.hpp"
#include "../../ops/embedding/op.hpp"
#include "../../ops/linear/op.hpp"
#include "../../ops/rms_norm/op.hpp"
#include "../../ops/rope/op.hpp"
#include "../../ops/self_attention/op.hpp"
#include "../../ops/swiglu/op.hpp"

#include <algorithm>
#include <cmath>
#include <cstring>

namespace llaisys::models {

void Qwen2Model::init_cache(llaisysDeviceType_t device, int device_id) {
    k_caches.resize(meta.nlayer);
    v_caches.resize(meta.nlayer);
    cache_len = 0;

    auto dev = device;
    auto did = device_id;

    q_flat = Tensor::create({meta.maxseq, meta.nh * meta.dh}, meta.dtype, dev, did);
    k_flat = Tensor::create({meta.maxseq, meta.nkvh * meta.dh}, meta.dtype, dev, did);
    v_flat = Tensor::create({meta.maxseq, meta.nkvh * meta.dh}, meta.dtype, dev, did);
    attn_out = Tensor::create({meta.maxseq, meta.nh, meta.dh}, meta.dtype, dev, did);
    attn_o_out = Tensor::create({meta.maxseq, meta.hs}, meta.dtype, dev, did);
    normed = Tensor::create({meta.maxseq, meta.hs}, meta.dtype, dev, did);
    normed2 = Tensor::create({meta.maxseq, meta.hs}, meta.dtype, dev, did);
    gate_out = Tensor::create({meta.maxseq, meta.di}, meta.dtype, dev, did);
    up_out = Tensor::create({meta.maxseq, meta.di}, meta.dtype, dev, did);
    swiglu_out = Tensor::create({meta.maxseq, meta.di}, meta.dtype, dev, did);
    mlp_out = Tensor::create({meta.maxseq, meta.hs}, meta.dtype, dev, did);
    logits = Tensor::create({meta.maxseq, meta.voc}, meta.dtype, dev, did);
    pos_ids = Tensor::create({meta.maxseq}, LLAISYS_DTYPE_I64, dev, did);
    max_idx = Tensor::create({1}, LLAISYS_DTYPE_I64, dev, did);
    max_val = Tensor::create({1}, meta.dtype, dev, did);
}

void Qwen2Model::forward(const int64_t *token_ids, size_t ntoken,
                         int64_t *out_token) {
    auto device = in_embed->deviceType();
    auto device_id = in_embed->deviceId();
    float scale = 1.0f / std::sqrt((float)meta.dh);
    size_t total_len = cache_len + ntoken;

    /* 把预分配的工作区张量裁剪成 ntoken 大小 */
    auto pos = pos_ids->slice(0, 0, ntoken);
    auto qf = q_flat->slice(0, 0, ntoken);
    auto kf = k_flat->slice(0, 0, ntoken);
    auto vf = v_flat->slice(0, 0, ntoken);
    auto ao = attn_out->slice(0, 0, ntoken);
    auto aoo = attn_o_out->slice(0, 0, ntoken);
    auto nm = normed->slice(0, 0, ntoken);
    auto nm2 = normed2->slice(0, 0, ntoken);
    auto go = gate_out->slice(0, 0, ntoken);
    auto uo = up_out->slice(0, 0, ntoken);
    auto so = swiglu_out->slice(0, 0, ntoken);
    auto mo = mlp_out->slice(0, 0, ntoken);
    auto lg = logits->slice(0, 0, ntoken);

    /* position ids */
    std::vector<int64_t> pos_vec(ntoken);
    for (size_t p = 0; p < ntoken; p++) {
        pos_vec[p] = (int64_t)(cache_len + p);
    }
    pos->load(pos_vec.data());

    /* 输入 token ids */
    auto idx_t = Tensor::create({ntoken}, LLAISYS_DTYPE_I64, device, device_id);
    idx_t->load(token_ids);

    /* 1. embedding */
    auto hidden = Tensor::create({ntoken, meta.hs}, meta.dtype, device, device_id);
    ops::embedding(hidden, idx_t, in_embed);

    /* 2. 逐层前向 */
    for (size_t l = 0; l < meta.nlayer; l++) {
        auto &L = layers[l];

        /* 2a. RMS Norm + Q/K/V */
        ops::rms_norm(nm, hidden, L.attn_norm_w, meta.epsilon);

        ops::linear(qf, nm, L.attn_q_w, L.attn_q_b);
        ops::linear(kf, nm, L.attn_k_w, L.attn_k_b);
        ops::linear(vf, nm, L.attn_v_w, L.attn_v_b);

        auto q = qf->view({ntoken, meta.nh, meta.dh});
        auto k = kf->view({ntoken, meta.nkvh, meta.dh});
        auto v = vf->view({ntoken, meta.nkvh, meta.dh});

        /* 2b. RoPE */
        ops::rope(q, q, pos, meta.theta);
        ops::rope(k, k, pos, meta.theta);

        /* 2c. 拼接 KV cache */
        auto full_k = Tensor::create({total_len, meta.nkvh, meta.dh},
                                     meta.dtype, device, device_id);
        auto full_v = Tensor::create({total_len, meta.nkvh, meta.dh},
                                     meta.dtype, device, device_id);

        if (cache_len > 0) {
            auto past_k_2d = k_caches[l]->view({cache_len, meta.nkvh * meta.dh});
            auto fk_start_2d = full_k->view({total_len, meta.nkvh * meta.dh})
                                   ->slice(0, 0, cache_len);
            auto fk_start = fk_start_2d->view({cache_len, meta.nkvh * meta.dh});
            fk_start->load(past_k_2d->data());

            auto past_v_2d = v_caches[l]->view({cache_len, meta.nkvh * meta.dh});
            auto fv_start_2d = full_v->view({total_len, meta.nkvh * meta.dh})
                                   ->slice(0, 0, cache_len);
            auto fv_start = fv_start_2d->view({cache_len, meta.nkvh * meta.dh});
            fv_start->load(past_v_2d->data());
        }

        {
            auto cur_k_2d = k->view({ntoken, meta.nkvh * meta.dh});
            auto fk_end_2d = full_k->view({total_len, meta.nkvh * meta.dh})
                                 ->slice(0, cache_len, total_len);
            auto fk_end = fk_end_2d->view({ntoken, meta.nkvh * meta.dh});
            fk_end->load(cur_k_2d->data());

            auto cur_v_2d = v->view({ntoken, meta.nkvh * meta.dh});
            auto fv_end_2d = full_v->view({total_len, meta.nkvh * meta.dh})
                                 ->slice(0, cache_len, total_len);
            auto fv_end = fv_end_2d->view({ntoken, meta.nkvh * meta.dh});
            fv_end->load(cur_v_2d->data());
        }

        k_caches[l] = full_k;
        v_caches[l] = full_v;

        /* 2d. SelfAttention */
        ops::self_attention(ao, q, full_k, full_v, scale);

        /* 2e. O 投影 + residual */
        auto attn_2d = ao->view({ntoken, meta.nh * meta.dh});
        ops::linear(aoo, attn_2d, L.attn_o_w, nullptr);
        ops::add(hidden, hidden, aoo);

        /* 2f. MLP */
        ops::rms_norm(nm2, hidden, L.mlp_norm_w, meta.epsilon);
        ops::linear(go, nm2, L.mlp_gate_w, nullptr);
        ops::linear(uo, nm2, L.mlp_up_w, nullptr);
        ops::swiglu(so, go, uo);
        ops::linear(mo, so, L.mlp_down_w, nullptr);
        ops::add(hidden, hidden, mo);
    }

    /* 3. 最终输出 */
    ops::rms_norm(nm, hidden, out_norm_w, meta.epsilon);
    ops::linear(lg, nm, out_embed, nullptr);

    /* 4. 取最后一个 token 的 argmax */
    auto last_logits_2d = lg->view({ntoken, meta.voc});
    auto last_row = last_logits_2d->slice(0, ntoken - 1, ntoken);
    auto last_1d = last_row->view({meta.voc});
    ops::argmax(max_idx, max_val, last_1d);

    int64_t next;
    std::memcpy(&next, max_idx->data(), sizeof(int64_t));
    *out_token = next;

    cache_len = total_len;
}

} // namespace llaisys::models
