# FlashAttention 文档 / Documentation

本目录包含 FlashAttention 的详细技术文档。
This directory contains detailed technical documentation for FlashAttention.

## 文档索引 / Document Index

### 核心文档 / Core Documentation

1. **[architecture.md](architecture.md)** - 架构概述 / Architecture Overview
   - FlashAttention 算法原理
   - 版本对比 (FA1/FA2/FA3)
   - 关键优化技术

2. **[core_files.md](core_files.md)** - 核心文件说明 / Core Files Description
   - Python 接口文件
   - C++/CUDA 内核文件
   - 文件组织结构

3. **[inference.md](inference.md)** - 推理运行逻辑 / Inference Runtime Logic
   - 推理使用指南
   - KV Cache 管理
   - 性能优化建议

4. **[algorithm.md](algorithm.md)** - 算法详解 / Algorithm Details
   - 前向传播算法
   - 反向传播算法
   - 数值稳定性

## 快速开始 / Quick Start

### FlashAttention 是什么?

FlashAttention 是一个快速且内存高效的精确注意力机制实现。通过分块计算和重计算策略,将注意力机制的内存复杂度从 O(N²) 降低到 O(N),同时保持精确的注意力计算结果。

FlashAttention is a fast and memory-efficient exact attention implementation. It reduces memory complexity from O(N²) to O(N) through tiling and recomputation, while maintaining exact attention computation.

### 主要特性 / Key Features

- **内存高效 / Memory Efficient**: O(N) 内存使用,而非标准实现的 O(N²)
- **速度快 / Fast**: 在现代 GPU 上比标准注意力快 2-4 倍
- **精确 / Exact**: 产生与标准注意力完全相同的结果
- **功能丰富 / Feature Rich**:
  - 因果掩码 (Causal Masking)
  - 滑动窗口注意力 (Sliding Window)
  - 变长序列 (Variable Length Sequences)
  - 多查询注意力 MQA/分组查询注意力 GQA
  - KV 缓存 (用于推理)
  - Dropout
  - 旋转位置编码 (RoPE)
  - ALiBi 位置偏置

### 支持的硬件 / Supported Hardware

- **NVIDIA GPUs**:
  - Ampere (SM80): A100, A10, RTX 30xx 系列
  - Hopper (SM90): H100, H800
  - Blackwell (SM100): B100, B200 (最新)

- **AMD GPUs** (实验性):
  - 通过 Triton 后端支持 ROCm

### 版本对比 / Version Comparison

| 版本 | 架构 | 主要特性 |
|------|------|----------|
| FlashAttention-1 | Ampere+ | 基础分块算法,降低内存使用 |
| FlashAttention-2 | Ampere+ | 优化的分块策略,减少非矩阵乘法操作 |
| FlashAttention-3 | Hopper+ | 使用 Cute-DSL,利用 TMA/GMMA 指令,支持更多特性 |

## 基本用法 / Basic Usage

```python
import torch
from flash_attn import flash_attn_func

# 准备输入张量 / Prepare input tensors
batch_size, seqlen, num_heads, head_dim = 4, 1024, 12, 64
q = torch.randn(batch_size, seqlen, num_heads, head_dim, dtype=torch.float16, device='cuda')
k = torch.randn(batch_size, seqlen, num_heads, head_dim, dtype=torch.float16, device='cuda')
v = torch.randn(batch_size, seqlen, num_heads, head_dim, dtype=torch.float16, device='cuda')

# 调用 FlashAttention / Call FlashAttention
output = flash_attn_func(q, k, v)
```

## 代码组织 / Code Organization

```
flash-attention/
├── flash_attn/              # Python 接口和模块
│   ├── flash_attn_interface.py    # 主要用户接口
│   ├── cute/                       # FlashAttention-3 实现 (Hopper/Blackwell)
│   │   ├── interface.py            # FA3 接口
│   │   ├── flash_fwd.py            # 前向内核
│   │   └── flash_bwd.py            # 反向内核
│   ├── modules/                    # PyTorch nn.Module 封装
│   └── flash_attn_triton_amd/      # AMD ROCm 后端
│
├── csrc/                    # C++/CUDA 源代码 (FlashAttention-1/2)
│   └── flash_attn/
│       ├── flash_api.cpp            # C++ API 入口
│       └── src/
│           ├── flash.h              # 参数结构定义
│           ├── flash_fwd_kernel.h   # 前向内核实现
│           └── flash_bwd_kernel.h   # 反向内核实现
│
├── hopper/                  # Hopper 架构优化实现
│   ├── flash_api.cpp
│   └── *.hpp                        # 内核头文件
│
└── docs/                    # 本文档目录
```

## 贡献者 / Contributors

FlashAttention 由以下研究者和工程师开发:
- Tri Dao (主要作者)
- Dan Fu
- Stefano Ermon
- Atri Rudra
- Christopher Ré
- Jay Shah, Ganesh Bikshandi, Ying Zhang (FlashAttention-3)

## 引用 / Citation

如果您在研究中使用 FlashAttention,请引用:

```bibtex
@inproceedings{dao2022flashattention,
  title={FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness},
  author={Dao, Tri and Fu, Daniel Y. and Ermon, Stefano and Rudra, Atri and R{\'e}, Christopher},
  booktitle={Advances in Neural Information Processing Systems},
  year={2022}
}

@inproceedings{dao2023flashattention2,
  title={FlashAttention-2: Faster Attention with Better Parallelism and Work Partitioning},
  author={Dao, Tri},
  booktitle={International Conference on Learning Representations},
  year={2024}
}
```

## 许可证 / License

请参考项目根目录的 LICENSE 文件。

## 支持 / Support

- GitHub Issues: https://github.com/Dao-AILab/flash-attention/issues
- 论文: https://arxiv.org/abs/2205.14135 (FA1), https://arxiv.org/abs/2307.08691 (FA2)
