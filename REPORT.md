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

- argmax — 1D 张量求最大值索引
- embedding — 查表取行
- linear — 矩阵乘法 Y = X·W^T + b
- rms_norm — RMS 归一化
- rope — 旋转位置编码
- self_attention — 因果自注意力（含 GQA 支持）
- swiglu — SwiGLU 激活函数

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

## 平台支持

| 平台 | 运行时 API | 算子 | 推理 |
|------|-----------|------|------|
| CPU (MinGW-w64) | ✅ 通过 | ✅ 通过 | 未完成 |
| NVIDIA (CUDA) | 未实现 | 未实现 | 未实现 |
