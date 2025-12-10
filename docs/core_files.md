# 核心文件说明 / Core Files Description

本文档描述 FlashAttention 代码库中的核心文件及其作用。

---

## 目录结构概览 / Directory Structure Overview

```
flash-attention/
├── flash_attn/              # Python 包和接口
├── csrc/                    # C++/CUDA 源代码 (FA1/FA2)
├── hopper/                  # Hopper 架构优化 (FA2/FA3)
├── tests/                   # 测试套件
├── benchmarks/              # 性能基准测试
└── docs/                    # 文档 (本目录)
```

---

## Python 接口层 / Python Interface Layer

### 1. `flash_attn/flash_attn_interface.py`

**作用**: FlashAttention 的主要 Python 接口

**关键内容**:
- `flash_attn_func()`: 标准注意力接口
- `flash_attn_qkvpacked_func()`: QKV 打包格式接口
- `flash_attn_kvpacked_func()`: KV 打包格式接口 (用于 MQA/GQA)
- `flash_attn_varlen_func()`: 变长序列接口
- `flash_attn_with_kvcache()`: KV 缓存接口 (推理)

**主要类**:
- `FlashAttnFunc`: 标准注意力的 autograd Function
- `FlashAttnQKVPackedFunc`: QKV 打包格式的 autograd Function
- `FlashAttnVarlenFunc`: 变长序列的 autograd Function

**使用示例**:
```python
from flash_attn import flash_attn_func

# 基本用法
output = flash_attn_func(q, k, v, dropout_p=0.1, causal=True)

# 带 softcap
output = flash_attn_func(q, k, v, softcap=30.0)

# 滑动窗口
output = flash_attn_func(q, k, v, window_size=(256, 256))
```

**关键函数注释**:
```python
def flash_attn_func(
    q, k, v,
    dropout_p=0.0,              # Dropout 概率
    softmax_scale=None,         # 缩放因子 (默认 1/√d)
    causal=False,               # 因果掩码
    window_size=(-1, -1),       # 滑动窗口 (左, 右)
    softcap=0.0,                # Softcap 值
    alibi_slopes=None,          # ALiBi 斜率
    deterministic=False,        # 确定性反向传播
    return_attn_probs=False,    # 是否返回注意力概率
):
    """
    Arguments:
        q: (batch, seqlen_q, nheads, headdim)
        k: (batch, seqlen_k, nheads_k, headdim)
        v: (batch, seqlen_k, nheads_k, headdim)

    Return:
        out: (batch, seqlen_q, nheads, headdim)
    """
```

---

### 2. `flash_attn/cute/interface.py`

**作用**: FlashAttention-3 的接口 (基于 Cute-DSL)

**支持的架构**:
- Hopper (SM90): H100, H800
- Blackwell (SM100): B100, B200

**新特性**:
- `score_mod`: 自定义分数修改函数
- `mask_mod`: 灵活的掩码函数
- `block_sparse_tensors`: 块稀疏注意力

**关键函数**:
```python
def flash_attn_func(
    q, k, v,
    # ... 标准参数 ...
    score_mod=None,           # 分数修改: (score, b, h, q_idx, kv_idx) -> score
    mask_mod=None,            # 掩码: (b, h, q_idx, kv_idx) -> bool
    block_sparse_tensors=None # 块稀疏张量
):
    """
    FlashAttention-3 接口,支持更多高级特性
    """
```

**使用示例**:
```python
# 自定义分数修改 (例如: 相对位置偏置)
def my_score_mod(score, batch_idx, head_idx, q_idx, kv_idx):
    relative_pos = q_idx - kv_idx
    return score + position_bias[relative_pos]

output = flash_attn_func(q, k, v, score_mod=my_score_mod)
```

---

### 3. `flash_attn/modules/mha.py`

**作用**: PyTorch `nn.Module` 封装

**主要类**:
- `FlashSelfAttention`: 自注意力模块
- `FlashCrossAttention`: 交叉注意力模块
- `FlashMHA`: 完整的多头注意力模块 (包含投影层)

**使用示例**:
```python
from flash_attn.modules.mha import FlashMHA

# 创建 MHA 模块
mha = FlashMHA(
    embed_dim=768,
    num_heads=12,
    dropout=0.1,
    causal=True
)

# 使用
output = mha(x)  # x: [batch, seqlen, embed_dim]
```

---

### 4. `flash_attn/cute/` 目录

FlashAttention-3 实现文件:

| 文件 | 作用 |
|------|------|
| `flash_fwd.py` | 前向内核 (Ampere/Hopper) |
| `flash_fwd_sm100.py` | 前向内核 (Blackwell) |
| `flash_bwd.py` | 反向内核 (Ampere) |
| `flash_bwd_sm90.py` | 反向内核 (Hopper) |
| `flash_bwd_sm100.py` | 反向内核 (Blackwell) |
| `softmax.py` | Softmax 实现 |
| `mask.py` | 掩码工具 |
| `tile_scheduler.py` | 分块调度器 |
| `block_sparse_utils.py` | 块稀疏工具 |

**代码组织**:
- 使用 Cute-DSL (NVIDIA CUTLASS DSL)
- 针对不同架构优化
- 支持更灵活的注意力变体

---

## C++/CUDA 内核层 / C++/CUDA Kernel Layer

### 1. `csrc/flash_attn/flash_api.cpp`

**作用**: C++ API 入口,连接 Python 和 CUDA 内核

**大小**: ~70,000 行 (包含大量模板实例化)

**主要函数**:
- `mha_fwd()`: 前向传播主函数
- `mha_varlen_fwd()`: 变长序列前向传播
- `mha_fwd_kvcache()`: KV 缓存前向传播
- `mha_bwd()`: 反向传播主函数

**功能**:
1. 参数验证和转换
2. 选择合适的内核模板
3. 配置 CUDA 启动参数
4. 调用对应的 CUDA 内核

**内核分发逻辑**:
```cpp
// 根据 head_dim 和 is_causal 选择模板
if (head_dim == 64) {
    if (is_causal) {
        run_mha_fwd_<cutlass::half_t, 64, true>(params, stream);
    } else {
        run_mha_fwd_<cutlass::half_t, 64, false>(params, stream);
    }
} else if (head_dim == 128) {
    // ...
}
```

---

### 2. `csrc/flash_attn/src/flash.h`

**作用**: 核心数据结构定义

**主要结构体**:

#### `Qkv_params`
基础参数结构,包含 Q、K、V 张量的指针和步长:
```cpp
struct Qkv_params {
    void *q_ptr, *k_ptr, *v_ptr;      // 数据指针
    index_t q_batch_stride, ...;       // 步长
    int h, h_k;                        // 头数
    int h_h_k_ratio;                   // MQA/GQA 比率
};
```

#### `Flash_fwd_params`
前向传播参数:
```cpp
struct Flash_fwd_params : public Qkv_params {
    void *o_ptr;                       // 输出
    void *softmax_lse_ptr;             // Log-sum-exp
    int b, seqlen_q, seqlen_k, d;      // 维度
    float scale_softmax;               // 缩放因子
    int *cu_seqlens_q, *cu_seqlens_k;  // 累积序列长度
    void *rotary_cos_ptr, *rotary_sin_ptr;  // RoPE
    int *block_table;                  // 页表 (paged KV)
    float p_dropout;                   // Dropout 概率
    int window_size_left, window_size_right;  // 滑动窗口
    float softcap;                     // Softcap
    // ... 更多参数
};
```

#### `Flash_bwd_params`
反向传播参数:
```cpp
struct Flash_bwd_params : public Flash_fwd_params {
    void *do_ptr;                      // 输出梯度
    void *dq_ptr, *dk_ptr, *dv_ptr;    // Q, K, V 梯度
    void *dsoftmax_sum;                // Softmax 梯度和
    bool deterministic;                // 确定性模式
};
```

---

### 3. `csrc/flash_attn/src/flash_fwd_kernel.h`

**作用**: 前向传播 CUDA 内核实现

**大小**: ~1000+ 行

**核心函数**: `compute_attn_1rowblock`

**算法流程**:
```cpp
template<typename Kernel_traits, bool Is_causal, ...>
__global__ void flash_fwd_kernel(...) {
    // 1. 加载 Q 块到共享内存
    // 2. 初始化输出和统计量
    // 3. 循环处理 K/V 块:
    for (int n_block = 0; n_block < n_block_max; ++n_block) {
        // 3.1 加载 K, V 块
        // 3.2 计算 S = Q @ K^T
        // 3.3 应用掩码和缩放
        // 3.4 在线 softmax 更新
        // 3.5 计算 O += P @ V
    }
    // 4. 写回输出和 LSE
}
```

**优化技术**:
- 使用 WMMA/MMA 指令进行矩阵乘法
- 流水线化内存访问和计算
- 寄存器分块减少共享内存使用
- 向量化加载/存储

---

### 4. `csrc/flash_attn/src/flash_bwd_kernel.h`

**作用**: 反向传播 CUDA 内核实现

**核心函数**: `compute_dq_dk_dv_1colblock`

**算法流程**:
```cpp
template<typename Kernel_traits, bool Is_causal, ...>
__global__ void flash_bwd_kernel(...) {
    // 1. 加载一个 K/V 块
    // 2. 初始化 dK, dV 累加器
    // 3. 循环处理 Q 块:
    for (int m_block = 0; m_block < m_block_max; ++m_block) {
        // 3.1 加载 Q, O, dO 块
        // 3.2 重计算 S = Q @ K^T
        // 3.3 重计算 P = softmax(S) (使用保存的 LSE)
        // 3.4 计算 dV += P^T @ dO
        // 3.5 计算 dP = dO @ V^T
        // 3.6 计算 dS = softmax_backward(dP, P)
        // 3.7 计算 dQ, dK
    }
    // 4. 写回梯度
}
```

**关键点**:
- **重计算**: 不加载 P,而是从 Q、K 重新计算
- **内存优势**: 避免存储 O(N²) 的注意力矩阵
- **数值稳定**: 使用保存的 LSE 确保 softmax 计算稳定

---

### 5. `csrc/flash_attn/src/` 其他关键文件

| 文件 | 作用 |
|------|------|
| `kernel_traits.h` | 内核配置 traits (块大小、线程数等) |
| `softmax.h` | 在线 softmax 算法实现 |
| `utils.h` | 工具函数 (reduce、类型转换等) |
| `mask.h` | 掩码函数 (因果、局部窗口) |
| `dropout.h` | Dropout 实现 |
| `rotary.h` | 旋转位置编码 (RoPE) |
| `alibi.h` | ALiBi 位置偏置 |
| `block_info.h` | 块索引计算 (varlen 支持) |
| `static_switch.h` | 编译期分支宏 |

---

### 6. `csrc/flash_attn/src/*.cu` 内核实例化文件

**示例**: `flash_fwd_hdim64_fp16_sm80.cu`

**作用**: 预编译不同配置的内核,减少编译时间

**命名规则**:
- `hdim64`: head_dim = 64
- `fp16`: 数据类型 float16
- `sm80`: 架构 (Ampere)
- `causal` (可选): 因果掩码版本

**总数**: 约 72 个实例化文件 (覆盖常用配置)

---

## Hopper 优化实现 / Hopper Optimized Implementation

### `hopper/` 目录结构

```
hopper/
├── flash_api.cpp              # Hopper API 入口
├── flash.h                    # 参数结构 (类似 csrc 版本,但有扩展)
├── mainloop_fwd_sm90_tma_gmma_ws.hpp   # 前向主循环 (使用 TMA/GMMA)
├── mainloop_bwd_sm90_tma_gmma_ws.hpp   # 反向主循环
├── tile_scheduler.hpp         # 高级分块调度
├── epilogue_fwd.hpp           # 前向输出处理
├── epilogue_bwd.hpp           # 反向输出处理
└── instantiations/            # 预编译实例 (450+ 个)
```

### 关键技术

#### 1. TMA (Tensor Memory Accelerator)
- 硬件加速的异步内存拷贝
- 从 HBM 直接到共享内存,无需经过寄存器
- 支持多维张量寻址

#### 2. GMMA (General Matrix Multiply-Accumulate)
- 新的矩阵乘法指令
- 比 WMMA 更快,更灵活
- 直接从共享内存读取

#### 3. Warp Specialization
- 不同 warp 执行不同任务:
  - Producer warp: 加载数据
  - Consumer warp: 计算
  - 流水线化,隐藏延迟

---

## 测试和基准测试 / Tests and Benchmarks

### `tests/` 目录

主要测试文件:
- `test_flash_attn.py`: 功能正确性测试
- `test_flash_attn_varlen.py`: 变长序列测试
- `test_flash_attn_causal.py`: 因果掩码测试

### `benchmarks/` 目录

性能基准测试:
- `benchmark_flash_attention.py`: 与标准实现对比
- `benchmark_forward.py`: 前向传播性能
- `benchmark_backward.py`: 反向传播性能

---

## 构建系统 / Build System

### `setup.py`

**作用**: 编译配置和安装脚本

**关键功能**:
1. 检测 GPU 架构
2. 设置 CUDA 编译标志
3. 选择要编译的内核
4. 支持 CUDA 和 ROCm

**编译选项**:
```bash
# 只编译特定架构
FLASH_ATTENTION_SKIP_CUDA_BUILD=TRUE pip install .

# 指定 CUDA 架构
TORCH_CUDA_ARCH_LIST="8.0;9.0" pip install .
```

---

## 文件依赖关系图 / File Dependency Graph

```
Python 层
  flash_attn_interface.py
           │
           ├─> flash_attn_2_cuda (CUDA 扩展)
           │         │
           │         └─> csrc/flash_attn/flash_api.cpp
           │                   │
           │                   ├─> flash_fwd_kernel.h
           │                   └─> flash_bwd_kernel.h
           │
           └─> cute/interface.py (FA3)
                     │
                     └─> hopper/flash_api.cpp
                               │
                               ├─> mainloop_fwd_sm90_tma_gmma_ws.hpp
                               └─> mainloop_bwd_sm90_tma_gmma_ws.hpp
```

---

## 快速定位指南 / Quick Reference Guide

### 如果你想...

| 目标 | 文件位置 |
|------|----------|
| 使用 FlashAttention | `flash_attn/flash_attn_interface.py` |
| 理解算法 | `csrc/flash_attn/src/flash_fwd_kernel.h` |
| 修改参数结构 | `csrc/flash_attn/src/flash.h` |
| 优化 Hopper 性能 | `hopper/mainloop_fwd_sm90_tma_gmma_ws.hpp` |
| 添加新特性 | `flash_attn/cute/interface.py` (Python), 对应的 .py 内核文件 |
| 调试精度问题 | `tests/test_flash_attn.py` |
| 性能调优 | `benchmarks/benchmark_flash_attention.py` |

---

## 代码风格和约定 / Code Style and Conventions

### 命名约定
- **Python**: `snake_case` for functions, `PascalCase` for classes
- **C++/CUDA**: `snake_case` for functions, `PascalCase` for types
- **模板参数**: `PascalCase` (e.g., `Kernel_traits`)

### 维度顺序
- **Batch-first**: `[batch, seqlen, num_heads, head_dim]`
- **与 PyTorch 一致**: 使得与其他 PyTorch 代码互操作简单

### 内存布局
- **默认**: Row-major (C 风格)
- **对齐**: Head_dim 必须是 8 的倍数 (向量化访问)

---

## 扩展开发指南 / Extension Development Guide

### 添加新的注意力变体

1. **Python 接口** (`flash_attn_interface.py`):
   ```python
   def flash_attn_my_variant(q, k, v, my_param):
       # 调用 CUDA 扩展
       return flash_attn_gpu.my_variant(q, k, v, my_param)
   ```

2. **C++ API** (`flash_api.cpp`):
   ```cpp
   std::vector<at::Tensor> mha_my_variant(...) {
       // 设置参数
       // 调用内核
   }
   ```

3. **CUDA 内核** (新建 `.cu` 文件):
   ```cpp
   template<...>
   __global__ void flash_my_variant_kernel(...) {
       // 实现
   }
   ```

4. **绑定** (在 `flash_api.cpp` 中):
   ```cpp
   PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
       m.def("my_variant", &mha_my_variant, ...);
   }
   ```

### 性能优化 Checklist

- [ ] 选择合适的块大小 (BLOCK_M, BLOCK_N)
- [ ] 最小化 HBM 访问
- [ ] 使用共享内存高效
- [ ] 向量化内存访问 (128-bit loads)
- [ ] 流水线化计算和内存访问
- [ ] 减少 warp 分歧
- [ ] 使用 async copy (TMA on Hopper)

---

## 常见问题 / FAQ

**Q: 为什么有这么多 .cu 文件?**
A: 预编译常用配置,加快编译和运行时内核选择。

**Q: csrc 和 hopper 有什么区别?**
A: csrc 是 Ampere 及更早架构的实现, hopper 专门为 Hopper (H100) 优化。

**Q: 如何选择使用哪个版本?**
A: 库会自动根据 GPU 架构选择最优实现。

**Q: 可以只编译我需要的内核吗?**
A: 可以,通过修改 `setup.py` 中的编译列表。

---

本文档持续更新中...
