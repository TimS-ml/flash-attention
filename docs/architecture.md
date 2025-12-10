# FlashAttention 架构文档 / Architecture Documentation

## 目录 / Table of Contents

1. [核心思想](#核心思想--core-idea)
2. [算法原理](#算法原理--algorithm-principles)
3. [版本演进](#版本演进--version-evolution)
4. [关键优化技术](#关键优化技术--key-optimizations)
5. [内存层次结构](#内存层次结构--memory-hierarchy)

---

## 核心思想 / Core Idea

### 标准注意力的问题 / Standard Attention Problems

标准的多头注意力机制 (Multi-Head Attention) 计算流程:

```
1. S = Q @ K^T              # 注意力分数矩阵, 形状: [batch, heads, seqlen_q, seqlen_k]
2. P = softmax(S / √d)      # 注意力概率矩阵
3. O = P @ V                # 输出矩阵
```

**主要问题**:
- **内存瓶颈**: 需要存储 S 和 P 矩阵,内存使用 O(batch × heads × N²)
- **带宽瓶颈**: 频繁的 HBM (高带宽内存) 访问导致性能受限
- **不可扩展**: 对于长序列 (N > 4096),内存需求快速增长

### FlashAttention 的解决方案 / FlashAttention Solution

**核心策略**: **分块计算 (Tiling) + 在线 Softmax + 重计算**

1. **分块计算**: 将 Q、K、V 分成小块,一次处理一个块
2. **在线 Softmax**: 增量计算 softmax,无需存储完整的注意力矩阵
3. **重计算**: 反向传播时重新计算注意力分数,而非存储

**内存优势**:
- 内存使用从 O(N²) 降低到 O(N)
- 中间结果存储在快速的 SRAM (片上内存),而非慢速的 HBM

---

## 算法原理 / Algorithm Principles

### 前向传播算法 / Forward Pass Algorithm

#### 伪代码 / Pseudocode

```python
# 输入: Q, K, V 的形状为 [batch, seqlen, heads, head_dim]
# 输出: O 的形状为 [batch, seqlen, heads, head_dim]

# 分块大小 (根据 SRAM 大小确定)
BLOCK_M = 128  # Query 块大小
BLOCK_N = 128  # Key/Value 块大小

# 将 Q 分成 Tr 块, K/V 分成 Tc 块
Tr = ceil(seqlen_q / BLOCK_M)
Tc = ceil(seqlen_k / BLOCK_N)

# 为每个 Q 块初始化输出和统计量
for i in range(Tr):
    # 从 HBM 加载 Q 块到 SRAM
    Qi = load_block(Q, i)  # [BLOCK_M, head_dim]

    # 初始化输出块和 softmax 统计量
    Oi = zeros([BLOCK_M, head_dim])
    li = zeros([BLOCK_M])  # 行的 logsumexp
    mi = -inf * ones([BLOCK_M])  # 行的最大值

    # 遍历所有 K/V 块
    for j in range(Tc):
        # 从 HBM 加载 K, V 块到 SRAM
        Kj = load_block(K, j)  # [BLOCK_N, head_dim]
        Vj = load_block(V, j)  # [BLOCK_N, head_dim]

        # 计算当前块的注意力分数
        Sij = Qi @ Kj.T / sqrt(head_dim)  # [BLOCK_M, BLOCK_N]

        # 应用掩码 (如果有)
        if causal:
            Sij = apply_causal_mask(Sij, i, j)

        # 在线 softmax 更新
        mi_new = max(mi, rowmax(Sij))
        Pij = exp(Sij - mi_new)

        # 更新 logsumexp
        li_new = exp(mi - mi_new) * li + rowsum(Pij)

        # 更新输出 (增量累加)
        Oi = diag(li / li_new) @ Oi + diag(1 / li_new) @ (Pij @ Vj)

        # 更新统计量
        li = li_new
        mi = mi_new

    # 将输出块写回 HBM
    store_block(O, i, Oi)
    store_block(LSE, i, mi + log(li))  # 保存 logsumexp 用于反向传播
```

#### 在线 Softmax 算法 / Online Softmax Algorithm

**关键思想**: 增量更新 softmax,无需存储完整的分数矩阵。

对于 softmax(x) = exp(x) / sum(exp(x)),我们维护:
- `m`: 当前最大值
- `l`: 当前 exp 和 (调整后的)

当看到新值 x_new 时:
```python
m_new = max(m, max(x_new))
l_new = l * exp(m - m_new) + sum(exp(x_new - m_new))
```

这允许我们分块处理数据,每次只看一小部分,但最终得到正确的 softmax 结果。

### 反向传播算法 / Backward Pass Algorithm

FlashAttention 的一个关键创新是**不存储注意力矩阵 P**,而是在反向传播时重新计算。

#### 策略 / Strategy

1. **保存**: 只保存 Q, K, V 和 logsumexp (LSE)
2. **重计算**: 在反向传播时重新计算 S = Q @ K^T 和 P = softmax(S)
3. **梯度计算**: 使用重计算的 P 和输入的 dO 计算 dQ, dK, dV

#### 内存权衡 / Memory Tradeoff

- **标准方法**: 保存 P (大小 O(N²)), 反向传播快但内存大
- **FlashAttention**: 不保存 P, 反向传播时重计算, 内存 O(N) 但稍慢
- **实际效果**: 由于避免了 HBM 访问,总体上反而更快!

---

## 版本演进 / Version Evolution

### FlashAttention-1 (2022)

**论文**: [FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness](https://arxiv.org/abs/2205.14135)

**主要贡献**:
- 首次提出分块注意力算法
- 在线 softmax 技术
- IO 感知的算法设计

**实现位置**: `csrc/flash_attn/`

**限制**:
- 分块策略不够优化
- 并行度有限

### FlashAttention-2 (2023)

**论文**: [FlashAttention-2: Faster Attention with Better Parallelism and Work Partitioning](https://arxiv.org/abs/2307.08691)

**改进**:
1. **减少非矩阵乘法操作**:
   - FA1: 在外层循环遍历 seqlen_k
   - FA2: 在外层循环遍历 seqlen_q, 减少写回次数

2. **更好的并行化**:
   - 在序列长度维度并行化 (而非只在 batch/heads)
   - 更好利用 GPU 的多个 SM (Streaming Multiprocessors)

3. **工作分区优化**:
   - 动态调整块大小
   - 减少 warp 之间的同步

**性能提升**: 比 FA1 快 2 倍左右

**实现位置**: `csrc/flash_attn/` (同一代码库,但算法不同)

### FlashAttention-3 (2024-2025)

**版本**: 基于 Cute-DSL 的实现,针对 Hopper 和 Blackwell 架构

**改进**:
1. **使用 Cute-DSL**: NVIDIA 的高级 CUDA 抽象层
2. **利用新硬件特性**:
   - **TMA** (Tensor Memory Accelerator): 异步内存拷贝
   - **GMMA** (General Matrix Multiply and Accumulate): 更快的矩阵乘法
   - **Warp Specialization**: 不同 warp 执行不同任务

3. **更多特性支持**:
   - `score_mod`: 自定义分数修改函数
   - `mask_mod`: 灵活的掩码函数
   - Block sparse attention: 块稀疏注意力
   - FP8 支持 (部分)

**实现位置**:
- `flash_attn/cute/` (Python 接口)
- `hopper/` (Hopper 优化的 C++)

**性能提升**: 在 H100 上比 FA2 快 1.5-2 倍

---

## 关键优化技术 / Key Optimizations

### 1. 内存层次结构优化 / Memory Hierarchy Optimization

现代 GPU 有多层内存:
```
SRAM (片上内存)    <- 快 (TB/s), 小 (~20MB)
    ↓
HBM (高带宽内存)   <- 慢 (1-2 TB/s), 大 (40-80GB)
```

**FlashAttention 策略**:
- 将工作数据保持在 SRAM 中
- 最小化 HBM 访问次数
- 批量读写 HBM (合并访问)

### 2. 分块策略 / Tiling Strategy

**块大小选择**:
- 取决于 head_dim, dropout, 架构等
- 典型值: BLOCK_M=128, BLOCK_N=64 或 128
- 目标: 最大化 SRAM 利用率,避免溢出

**二维分块**:
```
        K/V (seqlen_k)
        ┌─────┬─────┬─────┐
        │ K0  │ K1  │ K2  │
        └─────┴─────┴─────┘
Q       ┌─────┬─────┬─────┐
(seqlen)│ S00 │ S01 │ S02 │ Q0
        ├─────┼─────┼─────┤
        │ S10 │ S11 │ S12 │ Q1
        ├─────┼─────┼─────┤
        │ S20 │ S21 │ S22 │ Q2
        └─────┴─────┴─────┘
```

每个 thread block 处理一个 Q 块和所有 K/V 块的组合。

### 3. 因果掩码优化 / Causal Masking Optimization

对于因果注意力 (autoregressive 模型):
```
只需计算下三角部分:
┌───┬───┬───┬───┐
│ ✓ │ ✗ │ ✗ │ ✗ │
├───┼───┼───┼───┤
│ ✓ │ ✓ │ ✗ │ ✗ │
├───┼───┼───┼───┤
│ ✓ │ ✓ │ ✓ │ ✗ │
├───┼───┼───┼───┤
│ ✓ │ ✓ │ ✓ │ ✓ │
└───┴───┴───┴───┘
```

**优化**: 跳过完全被掩码的块,减少计算量。

### 4. 多查询/分组查询注意力 (MQA/GQA) / Multi-Query/Grouped-Query Attention

**标准 MHA**: 每个 query head 有自己的 key/value head
```
Q: [batch, seqlen, 12 heads, 64 dim]
K: [batch, seqlen, 12 heads, 64 dim]
V: [batch, seqlen, 12 heads, 64 dim]
```

**MQA** (Multi-Query Attention): 所有 query heads 共享一组 K/V
```
Q: [batch, seqlen, 12 heads, 64 dim]
K: [batch, seqlen, 1 head, 64 dim]   # 只有 1 个 head!
V: [batch, seqlen, 1 head, 64 dim]
```

**GQA** (Grouped-Query Attention): Query heads 分组共享 K/V
```
Q: [batch, seqlen, 12 heads, 64 dim]
K: [batch, seqlen, 3 heads, 64 dim]   # 4 个 Q heads 共享 1 个 KV head
V: [batch, seqlen, 3 heads, 64 dim]
```

**好处**: 减少 KV cache 大小,对推理特别重要。

### 5. KV 缓存管理 / KV Cache Management

对于自回归生成 (如 GPT):
- 每一步只生成一个新 token
- 之前的 K, V 可以复用
- 使用 KV cache 避免重复计算

**实现**:
```python
# 首次 prefill
cache_k = K  # [batch, seqlen_prompt, heads, dim]
cache_v = V

# 后续生成步骤
for step in range(max_new_tokens):
    q_new = model(tokens[-1])  # [batch, 1, heads, dim]
    k_new, v_new = ...         # [batch, 1, heads, dim]

    # 将新的 K/V 追加到缓存
    cache_k = cat([cache_k, k_new], dim=1)
    cache_v = cat([cache_v, v_new], dim=1)

    # 用完整的缓存计算注意力
    output = flash_attn_with_kvcache(
        q_new, cache_k, cache_v,
        k_new, v_new,  # 新的 K/V 用于更新缓存
        cache_seqlens=seqlen_prompt + step
    )
```

**Paged KV Cache**: 将缓存分成固定大小的页,使用页表索引,避免大块连续内存分配。

---

## 内存层次结构 / Memory Hierarchy

### GPU 内存架构 / GPU Memory Architecture

```
┌─────────────────────────────────────────┐
│  Registers (寄存器)                      │
│  - 最快                                  │
│  - 容量最小 (~256KB per SM)             │
│  - 每个线程私有                         │
└─────────────────────────────────────────┘
           ↓
┌─────────────────────────────────────────┐
│  Shared Memory / L1 Cache (SRAM)        │
│  - 非常快 (~19 TB/s on A100)            │
│  - 中等容量 (~192 KB per SM)            │
│  - Thread block 内共享                  │
└─────────────────────────────────────────┘
           ↓
┌─────────────────────────────────────────┐
│  L2 Cache                                │
│  - 快 (~5 TB/s)                         │
│  - 几 MB (40MB on A100)                 │
│  - 全局共享                             │
└─────────────────────────────────────────┘
           ↓
┌─────────────────────────────────────────┐
│  HBM (High Bandwidth Memory)             │
│  - 相对慢 (~1.5 TB/s on A100)           │
│  - 大容量 (40-80 GB)                    │
│  - 全局内存                             │
└─────────────────────────────────────────┘
```

### FlashAttention 内存访问模式 / Memory Access Pattern

```
HBM                    SRAM                    Compute
───                    ────                    ───────
Q  ─────────────────>  Qi    ─────────────>   Qi @ Kj^T
K  ─────────────────>  Kj                         ↓
V  ─────────────────>  Vj                      Softmax
                        ↓                          ↓
                       Oi  <──────────────    Pij @ Vj
                        ↓
O  <─────────────────  Oi (写回)
```

**关键点**:
1. Q, K, V 从 HBM 读入 SRAM (一次)
2. 所有计算在 SRAM 中完成
3. 输出写回 HBM (一次)
4. 中间结果 (S, P) 从不写入 HBM

### 算术强度分析 / Arithmetic Intensity

**算术强度** = 计算量 / 内存访问量

**标准注意力**:
- 计算: O(N² × d)
- 内存访问: O(N² + N × d)  (存储 S, P 矩阵)
- 算术强度: ~d (较低,内存瓶颈)

**FlashAttention**:
- 计算: O(N² × d)  (相同)
- 内存访问: O(N × d)  (只读写 Q, K, V, O)
- 算术强度: ~N (更高,计算瓶颈)

通过提高算术强度,FlashAttention 更好地利用 GPU 的计算能力。

---

## 性能特性 / Performance Characteristics

### 复杂度分析 / Complexity Analysis

| 指标 | 标准注意力 | FlashAttention |
|------|-----------|----------------|
| 时间复杂度 | O(N² × d) | O(N² × d) |
| 空间复杂度 | O(N² + N × d) | O(N × d) |
| HBM 访问 | O(N² + N × d) | O(N × d) |
| FLOPS 利用率 | ~30-40% | ~50-70% |

### 性能瓶颈 / Performance Bottlenecks

1. **短序列 (N < 512)**: 计算量小,启动开销占比大
2. **极大 head_dim (>256)**: 寄存器压力,可能溢出到 local memory
3. **高 dropout 率**: 需要存储 dropout mask

### 最佳使用场景 / Best Use Cases

- ✅ 长序列 (N > 1024)
- ✅ 训练 (需要反向传播)
- ✅ 批处理推理
- ✅ 因果/滑动窗口注意力
- ✅ MQA/GQA (减少 KV cache)

---

## 参考资料 / References

1. [FlashAttention论文](https://arxiv.org/abs/2205.14135)
2. [FlashAttention-2论文](https://arxiv.org/abs/2307.08691)
3. [Cute-DSL文档](https://github.com/NVIDIA/cutlass)
4. [GPU性能优化指南](https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/)
