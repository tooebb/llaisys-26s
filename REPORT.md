# LLAISYS 作业提交报告

## 环境

- OS: Windows 11
- 编译器: MinGW-w64 GCC 14.2.0
- 构建工具: Xmake v2.9.8
- Python: 3.13.7

## 完成情况

### 作业 #0：入门

- [x] 安装必备组件
- [x] Fork 并构建 LLAISYS
- [x] CPU 运行时测试通过 (`test/test_runtime.py --device cpu`)
- [x] 下载测试模型 (DeepSeek-R1-Distill-Qwen-1.5B)

### 作业 #1：张量

实现了以下函数 (`src/tensor/tensor.cpp`)：

- `load()` — 将主机数据加载到张量
- `isContiguous()` — 检查张量在内存中是否连续
- `view()` — 创建新张量，共享内存，改变形状
- `permute()` — 交换维度顺序
- `slice()` — 沿指定维度切片

测试结果：
```
python test/test_tensor.py  # 全部通过
```

### 作业 #2：算子

实现了 7 个 CPU 算子的 F32/F16/BF16 版本：

| 算子 | 功能 |
|------|------|
| argmax | 1D 张量求最大值索引 |
| embedding | 查表取行 |
| linear | 矩阵乘法 Y = X·W^T + b |
| rms_norm | RMS 归一化 |
| rope | 旋转位置编码 |
| self_attention | 因果自注意力（含 GQA 支持） |
| swiglu | SwiGLU 激活函数 |

测试结果：
```
python test/ops/argmax.py          # 通过
python test/ops/embedding.py       # 通过
python test/ops/linear.py          # 通过
python test/ops/rms_norm.py        # 通过
python test/ops/rope.py            # 通过
python test/ops/self_attention.py  # 通过
python test/ops/swiglu.py          # 通过
```

### 作业 #3：Qwen2 推理引擎

实现了完整的 Qwen2 LLM 推理流程：

| 模块 | 文件 | 说明 |
|------|------|------|
| C API 包装 | `src/llaisys/qwen2.cc` | 不透明句柄，4 个导出函数（Create/Destroy/Weights/Infer） |
| 模型结构 | `src/models/qwen2/model.{hpp,cpp}` | init_cache 预分配工作区，forward 完整前向流程 |
| ctypes 绑定 | `python/llaisys/libllaisys/models.py` | Structure 子类，argtypes/restype 设置 |
| Python 包装 | `python/llaisys/models/qwen2.py` | safetensors 权重加载 + generate() 自回归生成 |

前向流程：embedding → 28 层 (RMSNorm → Q/K/V 线性 → RoPE → KV Cache → SelfAttention → O 投影 → residual → RMSNorm → Gate/Up → SwiGLU → Down → residual) → RMSNorm → 输出线性 → argmax

测试结果：
```
python -c "from llaisys.models.qwen2 import Qwen2; ..."
# 输入 [1, 2, 3]，生成 5 个 token，结果 [4, 3, 3, 2, 3]，通过
```

## 复现流程

```bash
# 1. 环境准备
source setup_env.sh

# 2. 编译
xmake
xmake install
pip install ./python/

# 3. 运行测试
python test/test_tensor.py
python test/ops/argmax.py
python test/ops/embedding.py
python test/ops/linear.py
python test/ops/rms_norm.py
python test/ops/rope.py
python test/ops/self_attention.py
python test/ops/swiglu.py
```

### 作业 #4：CUDA 集成

实现了 NVIDIA CUDA 和沐曦 (Metax) MXMACA 两个平台的 GPU 加速代码。

| 模块 | 文件 |
|------|------|
| NVIDIA 运行时 API | `src/device/nvidia/nvidia_runtime_api.cu` — 12 个 CUDA Runtime 函数 |
| NVIDIA 资源 | `src/device/nvidia/nvidia_resource.cu/.cuh` — cuBLAS handle |
| Metax 运行时 API | `src/device/metax/metax_runtime_api.cu` — 12 个 MC Runtime 函数 |
| Metax 资源 | `src/device/metax/metax_resource.cu/.cuh` — mcBLAS handle |
| CUDA 算子 (×8) | `src/ops/*/nvidia/` — add, argmax, embedding, swiglu, rms_norm, rope, linear(cuBLAS), self_attention |
| MXMACA 算子 (×8) | `src/ops/*/metax/` — 同上，API 前缀 cuda→mc, cublas→mcblas |
| 编译配置 | `xmake/nvidia.lua`, `xmake/metax.lua` |
| 设备枚举 | `include/llaisys.h` — 新增 `LLAISYS_DEVICE_METAX = 2` |
| 分发逻辑 | 8 个 `op.cpp` — #ifdef ENABLE_NVIDIA_API / ENABLE_METAX_API |
| 模型适配 | `model.cpp` — device-aware memcpy (H2D/D2H) |

启用编译：
```bash
xmake f --nv-gpu=y --metax-gpu=y -cv && xmake
```

## 平台支持

| 平台 | 运行时 API | 算子 | 推理 |
|------|-----------|------|------|
| CPU (MinGW-w64) | ✅ 通过 | ✅ 通过 | ✅ 通过 |
| NVIDIA (CUDA) | ✅ 代码完成，缺卡未测 | ✅ 代码完成，缺卡未测 | ✅ 代码完成，缺卡未测 |
| Metax (MXMACA) | ✅ 代码完成，缺卡未测 | ✅ 代码完成，缺卡未测 | ✅ 代码完成，缺卡未测 |
