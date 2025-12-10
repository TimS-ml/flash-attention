# FlashAttention 算法详解 / Algorithm Details

本文档深入讲解 FlashAttention 的核心算法。

---

## 目录 / Table of Contents

1. [标准注意力机制](#标准注意力机制)
2. [在线 Softmax 算法](#在线-softmax-算法)
3. [前向传播算法](#前向传播算法)
4. [反向传播算法](#反向传播算法)
5. [复杂度分析](#复杂度分析)
6. [数值稳定性](#数值稳定性)

---

## 标准注意力机制 / Standard Attention Mechanism

### 数学定义

给定 Query (Q), Key (K), Value (V) 矩阵:

```
Q ∈ ℝ^(N×d)   # N 个查询向量,每个维度 d
K ∈ ℝ^(M×d)   # M 个键向量
V ∈ ℝ^(M×d_v) # M 个值向量,维度 d_v
```

注意力计算:

```
1. S = QK^T / √d          # 注意力分数, S ∈ ℝ^(N×M)
2. P = softmax(S)         # 注意力概率, P ∈ ℝ^(N×M)
3. O = PV                 # 输出, O ∈ ℝ^(N×d_v)
```

其中 softmax 按行计算:

```
P[i,j] = exp(S[i,j]) / Σ_k exp(S[i,k])
```

### 内存和计算复杂度

**内存**:
- 输入: Q, K, V = O(Nd + Md + Md_v)
- 中间结果: S, P = O(NM)  ← **瓶颈!**
- 输出: O = O(Nd_v)
- **总计**: O(NM + (N+M)d)

对于长序列 (N, M 很大), NM 项主导,导致 O(N²) 内存。

**计算 (FLOPs)**:
- S = QK^T: 2NMd (矩阵乘法)
- P = softmax(S): ~2NM (exp + 除法)
- O = PV: 2NMd_v
- **总计**: O(NMd)

### HBM 访问

**HBM 读取**:
- Q, K (计算 S): 2(N+M)d
- V (计算 O): Md_v
- S (计算 P): NM
- P (计算 O): NM

**HBM 写入**:
- S: NM
- P: NM  (如果保存用于反向传播)
- O: Nd_v

**总 HBM 访问**: O(NM + (N+M)d)

对于长序列,NM 项主导,导致**内存带宽瓶颈**。

---

## 在线 Softmax 算法 / Online Softmax Algorithm

这是 FlashAttention 的核心技术之一,允许分块计算 softmax 而无需一次看到所有数据。

### 问题

标准 softmax 需要两遍扫描:
```python
# 第一遍: 找最大值 (数值稳定性)
m = max(x)

# 第二遍: 计算 exp 和求和
s = sum(exp(x - m))

# 第三遍: 归一化
y = exp(x - m) / s
```

但在分块计算时,我们**不知道全局最大值**!

### 解决方案: 在线更新

维护运行中的统计量:
- `m`: 当前最大值
- `d`: 调整后的 exp 和

当看到新块 x' 时:

```python
# 更新最大值
m_new = max(m, max(x'))

# 更新 exp 和 (重要: 需要调整旧的和!)
d_new = d * exp(m - m_new) + sum(exp(x' - m_new))

# 更新统计量
m = m_new
d = d_new
```

**关键洞察**: 通过调整因子 `exp(m - m_new)`,我们可以"修正"之前计算的 exp 和,使其基于新的最大值。

### 数学推导

假设我们已经处理了块 x1, ..., x_{k-1},维护:
- m_{k-1} = max(x1, ..., x_{k-1})
- d_{k-1} = Σ_i exp(x_i - m_{k-1})

现在看到新块 x_k:

```
m_k = max(m_{k-1}, max(x_k))

d_k = Σ_{i=1}^{k} exp(x_i - m_k)
    = Σ_{i=1}^{k-1} exp(x_i - m_k) + Σ_j∈x_k exp(x_j - m_k)
    = Σ_{i=1}^{k-1} exp(x_i - m_{k-1} + m_{k-1} - m_k) + Σ_j∈x_k exp(x_j - m_k)
    = exp(m_{k-1} - m_k) * Σ_{i=1}^{k-1} exp(x_i - m_{k-1}) + Σ_j∈x_k exp(x_j - m_k)
    = exp(m_{k-1} - m_k) * d_{k-1} + Σ_j∈x_k exp(x_j - m_k)
```

因此:
```
d_k = d_{k-1} * exp(m_{k-1} - m_k) + sum(exp(x_k - m_k))
```

### 在注意力中的应用

对于注意力,我们还需要同时更新输出 O:

```python
# 初始化
m = -∞
d = 0
O = 0  # 累积输出

# 处理每个 KV 块
for j in range(num_blocks):
    # 计算当前块的分数
    S_j = Q @ K_j^T

    # 更新最大值
    m_new = max(m, rowmax(S_j))

    # 计算当前块的 exp
    P_j = exp(S_j - m_new)

    # 更新 exp 和
    d_new = d * exp(m - m_new) + rowsum(P_j)

    # 更新输出 (重要: 需要重新加权!)
    O = O * diag(d / d_new) * exp(m - m_new) + diag(1 / d_new) @ (P_j @ V_j)

    # 更新统计量
    m = m_new
    d = d_new

# 最终输出
return O
```

**关键**: 每次更新 m 时,我们需要**重新加权**之前累积的输出 O,因为之前的 softmax 概率基于旧的最大值。

---

## 前向传播算法 / Forward Pass Algorithm

### 完整算法 (FlashAttention-2)

```python
def flash_attention_forward(Q, K, V, block_M, block_N):
    """
    Args:
        Q: [N, d] query 矩阵
        K: [N, d] key 矩阵
        V: [N, d_v] value 矩阵
        block_M: Q 的块大小
        block_N: K/V 的块大小

    Returns:
        O: [N, d_v] 输出
        L: [N] log-sum-exp (用于反向传播)
    """
    N, d = Q.shape
    N, d_v = V.shape

    # 分块数量
    Tr = ceil(N / block_M)
    Tc = ceil(N / block_N)

    # 初始化输出
    O = zeros([N, d_v])
    L = zeros([N])  # LSE = log(sum(exp))

    # 处理每个 Q 块
    for i in range(Tr):
        # 加载 Q 块到 SRAM
        Qi = Q[i*block_M:(i+1)*block_M, :]  # [block_M, d]

        # 初始化输出和统计量
        Oi = zeros([block_M, d_v])
        li = zeros([block_M])  # exp 和
        mi = -inf * ones([block_M])  # 最大值

        # 处理每个 KV 块
        for j in range(Tc):
            # 加载 K, V 块到 SRAM
            Kj = K[j*block_N:(j+1)*block_N, :]  # [block_N, d]
            Vj = V[j*block_N:(j+1)*block_N, :]  # [block_N, d_v]

            # 计算注意力分数
            Sij = Qi @ Kj.T / sqrt(d)  # [block_M, block_N]

            # 应用因果掩码 (如果需要)
            if causal:
                # 块的全局行索引
                row_offset = i * block_M
                col_offset = j * block_N
                # 掩码: row < col 的位置
                mask = (row_offset + arange(block_M))[:, None] < (col_offset + arange(block_N))
                Sij[mask] = -inf

            # 在线 softmax 更新
            mi_new = max(mi, rowmax(Sij))  # [block_M]
            Pij = exp(Sij - mi_new[:, None])  # [block_M, block_N]

            # 更新 exp 和
            li_new = exp(mi - mi_new) * li + rowsum(Pij)  # [block_M]

            # 更新输出 (重新加权)
            scale_factor = exp(mi - mi_new) * (li / li_new)  # [block_M]
            Oi = diag(scale_factor) @ Oi + diag(1 / li_new) @ (Pij @ Vj)

            # 更新统计量
            li = li_new
            mi = mi_new

        # 写回输出和 LSE
        O[i*block_M:(i+1)*block_M, :] = Oi
        L[i*block_M:(i+1)*block_M] = mi + log(li)

    return O, L
```

### 内存访问分析

对于每个 Q 块 i:
- **读取**: Qi (一次), 所有 Kj, Vj (各一次)
- **写入**: Oi (一次)

**总 HBM 访问**:
```
读取: Tr * (block_M*d + Tc*(block_N*d + block_N*d_v))
    = Tr * block_M * d + Tr * Tc * block_N * (d + d_v)
    ≈ Nd + (N/block_M) * N * (d + d_v)  # 假设 block_M ≈ block_N

写入: N * d_v

总计: O(N*d + N²*(d+d_v)/block_M)
```

当 `block_M` 足够大 (≈ √N) 时,主导项为 O(Nd),而非标准方法的 O(N²)!

---

## 反向传播算法 / Backward Pass Algorithm

### 梯度公式

给定输出梯度 dO,需要计算:
- dQ: query 梯度
- dK: key 梯度
- dV: value 梯度

标准反向传播:

```
1. dV = P^T @ dO                # [M, N] @ [N, d_v] = [M, d_v]
2. dP = dO @ V^T                # [N, d_v] @ [d_v, M] = [N, M]
3. dS = softmax_backward(dP, P) # softmax 的反向传播
4. dQ = dS @ K / √d             # [N, M] @ [M, d] = [N, d]
5. dK = dS^T @ Q / √d           # [M, N] @ [N, d] = [M, d]
```

其中 softmax 的反向传播:

```python
def softmax_backward(dP, P):
    """
    dL/dS = P ⊙ (dL/dP - rowsum(dL/dP ⊙ P))
    """
    D = rowsum(dP * P)  # [N]
    dS = P * (dP - D[:, None])
    return dS
```

等价地,使用 D = rowsum(dO * O):

```
dS = P ⊙ (dP - D)
```

### FlashAttention 反向传播

**关键挑战**: 我们没有存储 P!

**解决方案**: 使用保存的 Q, K, V 和 LSE 重新计算 P。

```python
def flash_attention_backward(dO, Q, K, V, O, L, block_M, block_N):
    """
    Args:
        dO: [N, d_v] 输出梯度
        Q, K, V: [N, d] 或 [N, d_v] 输入
        O: [N, d_v] 前向传播的输出
        L: [N] 前向传播的 LSE

    Returns:
        dQ: [N, d] query 梯度
        dK: [N, d] key 梯度
        dV: [N, d_v] value 梯度
    """
    N, d = Q.shape
    N, d_v = V.shape

    # 初始化梯度
    dQ = zeros([N, d])
    dK = zeros([N, d])
    dV = zeros([N, d_v])

    # 预计算 D = rowsum(dO * O)
    D = rowsum(dO * O)  # [N]

    # 分块数量
    Tr = ceil(N / block_M)
    Tc = ceil(N / block_N)

    # 外层循环: 遍历 KV 块 (与前向传播不同!)
    for j in range(Tc):
        # 加载 K, V 块
        Kj = K[j*block_N:(j+1)*block_N, :]  # [block_N, d]
        Vj = V[j*block_N:(j+1)*block_N, :]  # [block_N, d_v]

        # 初始化 dK, dV 累加器
        dKj = zeros([block_N, d])
        dVj = zeros([block_N, d_v])

        # 内层循环: 遍历 Q 块
        for i in range(Tr):
            # 加载 Q, O, dO 块
            Qi = Q[i*block_M:(i+1)*block_M, :]      # [block_M, d]
            Oi = O[i*block_M:(i+1)*block_M, :]      # [block_M, d_v]
            dOi = dO[i*block_M:(i+1)*block_M, :]    # [block_M, d_v]
            Li = L[i*block_M:(i+1)*block_M]         # [block_M]
            Di = D[i*block_M:(i+1)*block_M]         # [block_M]

            # 重新计算 S 和 P
            Sij = Qi @ Kj.T / sqrt(d)  # [block_M, block_N]

            if causal:
                # 应用掩码
                row_offset = i * block_M
                col_offset = j * block_N
                mask = (row_offset + arange(block_M))[:, None] < (col_offset + arange(block_N))
                Sij[mask] = -inf

            # 从 LSE 重新计算 P
            # P = exp(S - L) where L = log(sum(exp(S)))
            Pij = exp(Sij - Li[:, None])  # [block_M, block_N]

            # 计算 dV
            dVj += Pij.T @ dOi  # [block_N, block_M] @ [block_M, d_v] = [block_N, d_v]

            # 计算 dP
            dPij = dOi @ Vj.T  # [block_M, d_v] @ [d_v, block_N] = [block_M, block_N]

            # Softmax 反向传播
            dSij = Pij * (dPij - Di[:, None])  # [block_M, block_N]

            # 计算 dQ (累加)
            dQi = dSij @ Kj / sqrt(d)  # [block_M, block_N] @ [block_N, d] = [block_M, d]
            dQ[i*block_M:(i+1)*block_M, :] += dQi

            # 计算 dK
            dKj += dSij.T @ Qi / sqrt(d)  # [block_N, block_M] @ [block_M, d] = [block_N, d]

        # 写回 dK, dV
        dK[j*block_N:(j+1)*block_N, :] = dKj
        dV[j*block_N:(j+1)*block_N, :] = dVj

    return dQ, dK, dV
```

### 为什么外层循环是 K/V?

在 FlashAttention-2 中,外层循环遍历 K/V 块,内层循环遍历 Q 块。这与前向传播相反!

**原因**:
- dK, dV 需要累加来自所有 Q 块的贡献
- 如果外层循环是 Q,我们需要多次读写 dK, dV (HBM 访问增加)
- 外层循环是 K/V 时,每个 dK_j, dV_j 只写回一次

---

## 复杂度分析 / Complexity Analysis

### 时间复杂度

| 操作 | 标准注意力 | FlashAttention |
|------|-----------|----------------|
| 前向传播 | O(N²d) | O(N²d) |
| 反向传播 | O(N²d) | O(N²d) |

**时间复杂度相同**,但 FlashAttention 实际运行更快,因为:
1. 更好的内存访问模式
2. 更高的算术强度 (计算/访存比)
3. 更好的缓存局部性

### 空间复杂度

| 内存 | 标准注意力 | FlashAttention |
|------|-----------|----------------|
| 前向传播 | O(N² + Nd) | O(Nd) |
| 反向传播 | O(N² + Nd) | O(Nd) |

**关键优势**: FlashAttention 避免存储 O(N²) 的注意力矩阵!

### HBM 访问复杂度

假设 SRAM 大小为 M,块大小 B = Θ(M^{1/2}):

| 操作 | 标准注意力 | FlashAttention |
|------|-----------|----------------|
| 前向传播 | O(Nd + N²) | O(N²d²/M) |
| 反向传播 | O(Nd + N²) | O(N²d²/M) |

当 M = Θ(d²) 时, FlashAttention 的 HBM 访问为 O(Nd),比标准方法的 O(N²) 好得多!

---

## 数值稳定性 / Numerical Stability

### Softmax 数值稳定性

标准 softmax 可能溢出:

```python
# 不稳定
exp(1000) = inf  # 溢出!
softmax([1000, 1001, 1002]) = [nan, nan, nan]
```

**解决方案**: 减去最大值

```python
# 稳定版本
def stable_softmax(x):
    m = max(x)
    exp_x = exp(x - m)
    return exp_x / sum(exp_x)
```

FlashAttention 使用在线 softmax,自动维护最大值 m,因此天然数值稳定。

### Log-Sum-Exp (LSE) 技巧

直接计算 log(sum(exp(x))) 可能溢出:

```python
# 不稳定
log(exp(1000) + exp(1001)) = log(inf) = inf
```

**稳定版本**:

```python
def logsumexp(x):
    m = max(x)
    return m + log(sum(exp(x - m)))
```

FlashAttention 保存 LSE = m + log(sum(exp(x - m))),用于:
1. 数值稳定性
2. 反向传播时重新计算 softmax

### FP16 vs BF16

**Float16 (FP16)**:
- 指数: 5 bits (范围 2^-14 到 2^15)
- 尾数: 10 bits
- 精度高但范围小

**BFloat16 (BF16)**:
- 指数: 8 bits (范围 2^-126 到 2^127, 与 FP32 相同!)
- 尾数: 7 bits
- 范围大但精度较低

**推荐**:
- Ampere (A100): 使用 FP16 (BF16 支持有限)
- Hopper (H100) 及更新: 使用 BF16 (数值稳定性更好)

### Softcap

对于非常大的 logits,softcap 可以防止数值问题:

```python
def softcap(x, cap):
    """
    限制 logits 的范围
    x -> cap * tanh(x / cap)
    """
    return cap * tanh(x / cap)
```

当 logits 很大时 (如 x = 100),softcap 将其限制在 [-cap, cap] 范围内。

---

## 优化技巧 / Optimization Tricks

### 1. 使用 exp2 而非 exp

在 CUDA 中,`exp2(x)` 比 `exp(x)` 快:

```cpp
// 慢
float y = expf(x);

// 快 (使用专用硬件指令)
float y = exp2f(x * 1.4426950408889634f);  // 1.4426... = 1/ln(2)
```

FlashAttention 内部使用 exp2 以提高性能。

### 2. 向量化访问

访问内存时,尽量使用 128-bit 向量化加载:

```cpp
// 慢: 4 次 32-bit 加载
float a = x[0];
float b = x[1];
float c = x[2];
float d = x[3];

// 快: 1 次 128-bit 加载
float4 abcd = *reinterpret_cast<float4*>(&x[0]);
```

这要求 head_dim 是 8 的倍数 (FP16 时 8*2 = 16 bytes = 128 bits)。

### 3. 寄存器分块

将小矩阵存储在寄存器而非共享内存:

```cpp
// Q 的一小块存储在寄存器
fragment<float, 16, 16> Q_frag;

// K, V 从共享内存加载,与 Q_frag 计算
```

寄存器访问比共享内存快 ~100 倍!

---

## 参考实现 / Reference Implementation

简化的 PyTorch 参考实现 (仅用于理解,非优化版):

```python
import torch
import math

def flash_attention_ref(Q, K, V, causal=False, block_size=64):
    """
    FlashAttention 的参考实现 (未优化,仅用于理解算法)

    Args:
        Q: [batch, seqlen_q, heads, dim]
        K: [batch, seqlen_k, heads, dim]
        V: [batch, seqlen_k, heads, dim_v]
        causal: 是否使用因果掩码
        block_size: 块大小

    Returns:
        O: [batch, seqlen_q, heads, dim_v]
    """
    B, N, H, D = Q.shape
    _, M, _, D_v = V.shape

    scale = 1.0 / math.sqrt(D)
    O = torch.zeros(B, N, H, D_v, device=Q.device, dtype=Q.dtype)
    L = torch.zeros(B, H, N, device=Q.device, dtype=torch.float32)

    num_q_blocks = (N + block_size - 1) // block_size
    num_kv_blocks = (M + block_size - 1) // block_size

    for i in range(num_q_blocks):
        q_start = i * block_size
        q_end = min((i + 1) * block_size, N)
        Qi = Q[:, q_start:q_end, :, :]  # [B, block_size, H, D]

        # 初始化
        Oi = torch.zeros(B, q_end - q_start, H, D_v, device=Q.device, dtype=Q.dtype)
        li = torch.zeros(B, H, q_end - q_start, device=Q.device, dtype=torch.float32)
        mi = torch.full((B, H, q_end - q_start), float('-inf'), device=Q.device, dtype=torch.float32)

        for j in range(num_kv_blocks):
            kv_start = j * block_size
            kv_end = min((j + 1) * block_size, M)
            Kj = K[:, kv_start:kv_end, :, :]  # [B, block_size, H, D]
            Vj = V[:, kv_start:kv_end, :, :]  # [B, block_size, H, D_v]

            # S = Q @ K^T
            Sij = torch.einsum('bqhd,bkhd->bhqk', Qi, Kj) * scale  # [B, H, q_len, k_len]

            # 因果掩码
            if causal:
                mask = torch.arange(q_start, q_end, device=Q.device)[:, None] < torch.arange(kv_start, kv_end, device=Q.device)
                Sij = Sij.masked_fill(mask, float('-inf'))

            # 在线 softmax
            mi_new = torch.maximum(mi, Sij.max(dim=-1)[0])  # [B, H, q_len]
            Pij = torch.exp(Sij - mi_new.unsqueeze(-1))  # [B, H, q_len, k_len]
            li_new = li * torch.exp(mi - mi_new) + Pij.sum(dim=-1)  # [B, H, q_len]

            # 更新输出
            scale_factor = (torch.exp(mi - mi_new) * (li / li_new)).unsqueeze(-1)  # [B, H, q_len, 1]
            Oi = Oi * scale_factor.transpose(1, 2) + torch.einsum('bhqk,bkhd->bqhd', Pij / li_new.unsqueeze(-1), Vj)

            mi = mi_new
            li = li_new

        O[:, q_start:q_end, :, :] = Oi
        L[:, :, q_start:q_end] = mi + torch.log(li)

    return O, L
```

---

本文档持续更新中...

## 推荐阅读 / Recommended Reading

1. [FlashAttention 论文](https://arxiv.org/abs/2205.14135)
2. [Online Normalizer Calculation for Softmax](https://arxiv.org/abs/1805.02867)
3. [Self-attention Does Not Need O(n²) Memory](https://arxiv.org/abs/2112.05682)
