import ctypes
from ctypes import c_int
import json
import re
from pathlib import Path
from typing import Sequence

import safetensors

from ..libllaisys import (
    LIB_LLAISYS,
    DeviceType,
    DataType,
    LlaisysQwen2Meta,
    LlaisysQwen2Weights,
)

# config.json 中的 dtype 字符串 -> LLAISYS DataType
_DTYPE_MAP = {
    "float32": DataType.F32,
    "float16": DataType.F16,
    "bfloat16": DataType.BF16,
}


def _load_config(model_path: Path):
    config_path = model_path / "config.json"
    with open(config_path, "r", encoding="utf-8") as f:
        return json.load(f)


def _make_meta(cfg: dict) -> LlaisysQwen2Meta:
    hs = cfg["hidden_size"]
    nh = cfg["num_attention_heads"]
    nkvh = cfg.get("num_key_value_heads", nh)
    dh = cfg.get("head_dim", hs // nh)
    dtype_str = cfg.get("torch_dtype", "float32")

    meta = LlaisysQwen2Meta()
    meta.dtype = _DTYPE_MAP.get(dtype_str, DataType.F32)
    meta.nlayer = cfg["num_hidden_layers"]
    meta.hs = hs
    meta.nh = nh
    meta.nkvh = nkvh
    meta.dh = dh
    meta.di = cfg["intermediate_size"]
    # 限制最大序列长度，避免分配过多内存
    maxseq = cfg.get("max_position_embeddings", 2048)
    meta.maxseq = min(maxseq, 2048)
    meta.voc = cfg["vocab_size"]
    meta.epsilon = cfg.get("rms_norm_eps", 1e-6)
    meta.theta = cfg.get("rope_theta", 10000.0)
    meta.end_token = cfg.get("eos_token_id", -1)
    return meta


def _load_tensor_data(tensor_handle, torch_tensor):
    """把 PyTorch tensor 的数据拷进 C tensor"""
    # 确保是 contiguous 的
    t = torch_tensor.contiguous()
    ptr = t.data_ptr()
    LIB_LLAISYS.tensorLoad(tensor_handle, ctypes.c_void_p(ptr))


def _safe_get_tensor(data, keys):
    """从 safetensors 数据中按优先级取 key，返回第一个匹配的"""
    if isinstance(keys, str):
        keys = [keys]
    for k in keys:
        if k in data:
            return data[k]
    return None


class Qwen2:
    def __init__(self, model_path: str, device: DeviceType = DeviceType.CPU):
        model_path = Path(model_path)
        cfg = _load_config(model_path)
        self._meta = _make_meta(cfg)
        self._device = device
        self._end_token = int(self._meta.end_token)

        # 创建模型
        device_ids = (c_int * 1)(0)
        self._handle = LIB_LLAISYS.llaisysQwen2ModelCreate(
            ctypes.byref(self._meta),
            device,
            device_ids,
            1,
        )
        if not self._handle:
            raise RuntimeError("llaisysQwen2ModelCreate 返回空")

        # 获取权重结构
        weights_ptr = LIB_LLAISYS.llaisysQwen2ModelWeights(self._handle)
        self._weights = weights_ptr.contents

        # 加载 safetensors 权重
        self._load_all_weights(model_path, self._meta.nlayer)

    def _load_all_weights(self, model_path: Path, nlayer: int):
        """读取所有 .safetensors 文件并填入权重"""
        # 收集所有 tensor 数据 (用 PyTorch, 因为 bfloat16 numpy 不支持)
        all_data = {}
        for f in sorted(model_path.glob("*.safetensors")):
            with safetensors.safe_open(str(f), framework="pt", device="cpu") as sf:
                for key in sf.keys():
                    all_data[key] = sf.get_tensor(key)

        # 全局权重
        _load_tensor_data(self._weights.in_embed,
                          _safe_get_tensor(all_data, "model.embed_tokens.weight"))

        out_embed_data = _safe_get_tensor(all_data, "lm_head.weight")
        if out_embed_data is not None:
            _load_tensor_data(self._weights.out_embed, out_embed_data)
        else:
            # tied embedding: 用 in_embed 的数据
            _load_tensor_data(self._weights.out_embed,
                              all_data["model.embed_tokens.weight"])

        _load_tensor_data(self._weights.out_norm_w,
                          _safe_get_tensor(all_data, "model.norm.weight"))

        #每层权重
        for l in range(nlayer):
            prefix = f"model.layers.{l}."

            _load_tensor_data(
                self._weights.attn_norm_w[l],
                all_data[f"{prefix}input_layernorm.weight"])

            _load_tensor_data(
                self._weights.attn_q_w[l],
                all_data[f"{prefix}self_attn.q_proj.weight"])
            _load_tensor_data(
                self._weights.attn_q_b[l],
                all_data[f"{prefix}self_attn.q_proj.bias"])

            _load_tensor_data(
                self._weights.attn_k_w[l],
                all_data[f"{prefix}self_attn.k_proj.weight"])
            _load_tensor_data(
                self._weights.attn_k_b[l],
                all_data[f"{prefix}self_attn.k_proj.bias"])

            _load_tensor_data(
                self._weights.attn_v_w[l],
                all_data[f"{prefix}self_attn.v_proj.weight"])
            _load_tensor_data(
                self._weights.attn_v_b[l],
                all_data[f"{prefix}self_attn.v_proj.bias"])

            _load_tensor_data(
                self._weights.attn_o_w[l],
                all_data[f"{prefix}self_attn.o_proj.weight"])

            _load_tensor_data(
                self._weights.mlp_norm_w[l],
                all_data[f"{prefix}post_attention_layernorm.weight"])

            _load_tensor_data(
                self._weights.mlp_gate_w[l],
                all_data[f"{prefix}mlp.gate_proj.weight"])
            _load_tensor_data(
                self._weights.mlp_up_w[l],
                all_data[f"{prefix}mlp.up_proj.weight"])
            _load_tensor_data(
                self._weights.mlp_down_w[l],
                all_data[f"{prefix}mlp.down_proj.weight"])

    def generate(
        self,
        inputs: Sequence[int],
        max_new_tokens: int = None,
        top_k: int = 1,
        top_p: float = 0.8,
        temperature: float = 0.8,
    ):
        if max_new_tokens is None:
            max_new_tokens = 64

        token_ids = list(inputs)
        n_input = len(token_ids)

        # prefill: 所有输入 token 一起处理
        arr = (ctypes.c_int64 * n_input)(*token_ids)
        next_token = LIB_LLAISYS.llaisysQwen2ModelInfer(
            self._handle, arr, n_input)

        generated = [next_token]

        # 逐 token 生成
        for _ in range(max_new_tokens - 1):
            if next_token == self._end_token:
                break
            one = (ctypes.c_int64 * 1)(next_token)
            next_token = LIB_LLAISYS.llaisysQwen2ModelInfer(
                self._handle, one, 1)
            generated.append(next_token)

        return generated

    def __del__(self):
        if hasattr(self, "_handle") and self._handle:
            LIB_LLAISYS.llaisysQwen2ModelDestroy(self._handle)
            self._handle = None