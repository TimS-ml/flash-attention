# FlashAttention 推理运行逻辑 / Inference Runtime Logic

本文档详细介绍如何在推理场景中使用 FlashAttention,特别是 LLM (大语言模型) 推理。

---

## 目录 / Table of Contents

1. [推理 vs 训练](#推理-vs-训练)
2. [KV Cache 基础](#kv-cache-基础)
3. [基本推理示例](#基本推理示例)
4. [高级特性](#高级特性)
5. [性能优化](#性能优化)
6. [常见模式](#常见模式)

---

## 推理 vs 训练 / Inference vs Training

### 训练场景 / Training Scenario

```python
# 训练: 处理完整序列,需要反向传播
q, k, v = ... # [batch, seqlen, heads, dim]
output = flash_attn_func(q, k, v, causal=True)
loss = compute_loss(output, labels)
loss.backward()  # 需要梯度
```

**特点**:
- 完整序列一次性处理
- 需要计算梯度
- 批量大,序列长度固定
- 重视吞吐量

### 推理场景 / Inference Scenario

```python
# 推理: 逐 token 生成,无需梯度
with torch.no_grad():
    for step in range(max_new_tokens):
        q_new = ... # [batch, 1, heads, dim]  # 只有新 token!
        output = flash_attn_with_kvcache(
            q_new, cache_k, cache_v,
            k_new, v_new,
            cache_seqlens=current_len
        )
```

**特点**:
- 自回归生成 (一次一个 token)
- 无需梯度
- 批量小,序列长度动态增长
- 重视延迟

---

## KV Cache 基础 / KV Cache Basics

### 为什么需要 KV Cache?

自回归生成中,每一步只生成一个新 token,但注意力需要看到所有之前的 tokens:

```
步骤 1: "The"           -> attend to ["The"]
步骤 2: "The cat"       -> attend to ["The", "cat"]
步骤 3: "The cat sat"   -> attend to ["The", "cat", "sat"]
...
```

如果每次都重新计算所有 K, V,会**非常浪费**:

```python
# 低效方法 (每步重新计算所有 K, V)
for step in range(100):
    all_tokens = tokens[:step+1]  # 越来越长!
    q, k, v = model(all_tokens)   # 重复计算之前的 tokens
    output = flash_attn_func(q[-1:], k, v)  # 只需要最后一个 q
```

**KV Cache 解决方案**: 缓存之前步骤的 K, V,只计算新 token 的 K, V:

```python
# 高效方法 (使用 KV Cache)
cache_k, cache_v = [], []

for step in range(100):
    new_token = tokens[step]
    q_new, k_new, v_new = model(new_token)  # 只处理新 token!

    # 追加到缓存
    cache_k.append(k_new)
    cache_v.append(v_new)

    # 注意力使用完整缓存
    output = flash_attn_with_kvcache(
        q_new,
        torch.cat(cache_k, dim=1),  # 所有历史 K
        torch.cat(cache_v, dim=1),  # 所有历史 V
        k_new, v_new,
        cache_seqlens=step
    )
```

---

## 基本推理示例 / Basic Inference Example

### 1. Prefill 阶段 (处理提示词)

```python
import torch
from flash_attn import flash_attn_func, flash_attn_with_kvcache

# 配置
batch_size = 2
num_heads = 32
head_dim = 128
max_seq_len = 2048

# 1. Prefill: 处理输入提示词
prompt = "Once upon a time"
prompt_tokens = tokenizer.encode(prompt)  # [7] (假设 7 个 tokens)
prompt_len = len(prompt_tokens)

# 通过模型获取 Q, K, V
with torch.no_grad():
    hidden = model.embed(prompt_tokens)  # [batch=1, seqlen=7, hidden_dim]

    # 计算初始的 Q, K, V
    q = model.q_proj(hidden)  # [1, 7, num_heads, head_dim]
    k = model.k_proj(hidden)
    v = model.v_proj(hidden)

    # 第一次注意力计算 (使用标准接口)
    attn_output = flash_attn_func(
        q, k, v,
        causal=True,  # 因果掩码
        softmax_scale=1.0 / math.sqrt(head_dim)
    )  # [1, 7, num_heads, head_dim]

    # 初始化 KV cache
    cache_k = k  # [1, 7, num_heads, head_dim]
    cache_v = v
    cache_seqlen = prompt_len

# 继续模型前向传播...
output = model.output_proj(attn_output)
```

### 2. Decode 阶段 (生成新 tokens)

```python
# 2. Decode: 逐个生成新 tokens
max_new_tokens = 100
generated_tokens = []

for step in range(max_new_tokens):
    # 2.1 从 logits 采样新 token
    logits = model.lm_head(output[:, -1, :])  # 只需要最后一个位置
    next_token = sample(logits)  # 采样 (greedy/top-k/top-p)
    generated_tokens.append(next_token)

    # 2.2 计算新 token 的 Q, K, V
    with torch.no_grad():
        hidden_new = model.embed(next_token)  # [1, 1, hidden_dim]
        q_new = model.q_proj(hidden_new)      # [1, 1, num_heads, head_dim]
        k_new = model.k_proj(hidden_new)
        v_new = model.v_proj(hidden_new)

        # 2.3 使用 KV Cache 进行注意力计算
        attn_output = flash_attn_with_kvcache(
            q=q_new,                    # [1, 1, num_heads, head_dim]
            k_cache=cache_k,            # [1, cache_seqlen, num_heads, head_dim]
            v_cache=cache_v,            # [1, cache_seqlen, num_heads, head_dim]
            k=k_new,                    # 新的 K (会追加到 cache_k)
            v=v_new,                    # 新的 V (会追加到 cache_v)
            cache_seqlens=cache_seqlen, # 当前缓存长度
            causal=True,
            softmax_scale=1.0 / math.sqrt(head_dim)
        )  # [1, 1, num_heads, head_dim]

    # 2.4 更新缓存长度
    cache_seqlen += 1

    # 2.5 继续前向传播
    output = model.output_proj(attn_output)

    # 2.6 停止条件
    if next_token == eos_token_id:
        break

# 生成完成
generated_text = tokenizer.decode(generated_tokens)
```

---

## 高级特性 / Advanced Features

### 1. 预分配 KV Cache (推荐)

为了避免频繁的内存分配和拷贝,推荐预分配 KV Cache:

```python
# 预分配 KV cache (最大序列长度)
max_seq_len = 2048
cache_k = torch.zeros(
    batch_size, max_seq_len, num_heads, head_dim,
    dtype=torch.float16, device='cuda'
)
cache_v = torch.zeros(
    batch_size, max_seq_len, num_heads, head_dim,
    dtype=torch.float16, device='cuda'
)

# Prefill 阶段: 填充缓存的前 prompt_len 位置
cache_k[:, :prompt_len, :, :] = k_prefill
cache_v[:, :prompt_len, :, :] = v_prefill
cache_seqlen = prompt_len

# Decode 阶段: 追加新 K/V
for step in range(max_new_tokens):
    # ... 计算 k_new, v_new ...

    # 使用 cache_seqlens 参数指定追加位置
    attn_output = flash_attn_with_kvcache(
        q=q_new,
        k_cache=cache_k,
        v_cache=cache_v,
        k=k_new,         # 会被写入 cache_k[:, cache_seqlen, :, :]
        v=v_new,         # 会被写入 cache_v[:, cache_seqlen, :, :]
        cache_seqlens=cache_seqlen,  # 追加位置
        causal=True
    )

    cache_seqlen += 1
```

### 2. 批量推理 (Batching)

处理多个请求以提高吞吐量:

```python
# 多个请求,不同长度
batch_size = 4
prompts = [
    "Once upon a time",      # 长度 7
    "Hello world",           # 长度 2
    "The quick brown fox",   # 长度 4
    "In the beginning"       # 长度 3
]

# 方法 1: Padding (简单但浪费)
max_len = max(len(p) for p in prompts)
padded_prompts = [p + [pad_token] * (max_len - len(p)) for p in prompts]

# 方法 2: Variable-length (高效)
# 使用 flash_attn_varlen_func 和 cu_seqlens
```

### 3. 变长序列批处理 (Varlen Batching)

避免 padding 浪费:

```python
from flash_attn import flash_attn_varlen_func

# 将批次中的所有 tokens 打包成一个序列
# 例如: ["Hello world", "Hi"] -> "HelloworldHi"
all_tokens = torch.cat([encode(p) for p in prompts], dim=0)  # [total_tokens]

# 累积序列长度
seqlens = [len(encode(p)) for p in prompts]  # [2, 2]
cu_seqlens = torch.tensor([0] + list(cumsum(seqlens)), dtype=torch.int32)
# cu_seqlens = [0, 2, 4]  # 表示: [0:2], [2:4]

# Prefill
hidden = model.embed(all_tokens)  # [total_tokens, hidden_dim]
q = model.q_proj(hidden)          # [total_tokens, num_heads, head_dim]
k = model.k_proj(hidden)
v = model.v_proj(hidden)

attn_output = flash_attn_varlen_func(
    q, k, v,
    cu_seqlens_q=cu_seqlens,
    cu_seqlens_k=cu_seqlens,
    max_seqlen_q=max(seqlens),
    max_seqlen_k=max(seqlens),
    causal=True
)

# Decode 阶段类似,但每步只有 batch_size 个新 tokens
```

### 4. Paged KV Cache (PagedAttention)

对于非常长的序列或大批量,使用分页缓存:

```python
# 分页参数
page_size = 256  # 每页 256 个 tokens
num_pages = 1000  # 预分配 1000 页

# 分页 KV cache: [num_pages, page_size, num_heads, head_dim]
paged_k_cache = torch.zeros(
    num_pages, page_size, num_heads, head_dim,
    dtype=torch.float16, device='cuda'
)
paged_v_cache = torch.zeros(
    num_pages, page_size, num_heads, head_dim,
    dtype=torch.float16, device='cuda'
)

# 页表: [batch_size, max_num_pages_per_seq]
# 将逻辑序列位置映射到物理页
block_table = torch.zeros(
    batch_size, max_num_pages_per_seq,
    dtype=torch.int32, device='cuda'
)

# 为每个序列分配页
# 例如: 序列 0 使用页 [0, 1, 2]
block_table[0, :3] = torch.tensor([0, 1, 2])

# 使用分页 KV cache
attn_output = flash_attn_with_kvcache(
    q=q_new,
    k_cache=paged_k_cache,  # 分页的!
    v_cache=paged_v_cache,
    k=k_new,
    v=v_new,
    cache_seqlens=cache_seqlens,
    block_table=block_table,  # 提供页表
    causal=True
)
```

**优势**:
- 无需为每个序列分配连续内存
- 可以动态分配/释放页
- 更好的内存利用率 (避免碎片)
- 支持 longer context (>100k tokens)

### 5. 旋转位置编码 (RoPE)

许多现代 LLM (如 LLaMA) 使用 RoPE:

```python
# 预计算 RoPE 的 cos/sin
def precompute_rope(dim, max_seq_len, theta=10000.0):
    inv_freq = 1.0 / (theta ** (torch.arange(0, dim, 2).float() / dim))
    t = torch.arange(max_seq_len).float()
    freqs = torch.outer(t, inv_freq)  # [max_seq_len, dim/2]
    cos = freqs.cos()  # [max_seq_len, dim/2]
    sin = freqs.sin()
    return cos, sin

rotary_cos, rotary_sin = precompute_rope(head_dim, max_seq_len)
rotary_cos = rotary_cos.to(device='cuda', dtype=torch.float16)
rotary_sin = rotary_sin.to(device='cuda', dtype=torch.float16)

# 在 flash_attn_with_kvcache 中使用
attn_output = flash_attn_with_kvcache(
    q=q_new,
    k_cache=cache_k,
    v_cache=cache_v,
    k=k_new,
    v=v_new,
    rotary_cos=rotary_cos,  # [max_seq_len, head_dim/2]
    rotary_sin=rotary_sin,
    cache_seqlens=cache_seqlen,
    rotary_interleaved=True,  # 或 False (GPT-NeoX style)
    causal=True
)
```

FlashAttention 会**自动**在内核内部应用 RoPE,无需预先旋转 Q/K!

### 6. 多查询/分组查询注意力 (MQA/GQA)

减少 KV cache 大小:

```python
# GQA 示例: 32 个 Q heads, 8 个 KV heads (4:1 ratio)
num_heads_q = 32
num_heads_kv = 8
head_dim = 128

# Q 投影
q = q_proj(hidden)  # [batch, seqlen, num_heads_q=32, head_dim]

# K, V 投影 (更少的 heads!)
k = k_proj(hidden)  # [batch, seqlen, num_heads_kv=8, head_dim]
v = v_proj(hidden)  # [batch, seqlen, num_heads_kv=8, head_dim]

# FlashAttention 自动处理 GQA
# 每 4 个 Q heads 共享 1 个 KV head
attn_output = flash_attn_func(q, k, v, causal=True)
# 输出: [batch, seqlen, num_heads_q=32, head_dim]

# KV cache 大小减少 4 倍!
```

---

## 性能优化 / Performance Optimization

### 1. 内存优化

#### 使用 FP16/BF16

```python
# 使用半精度减少内存和带宽
q = q.to(torch.float16)  # 或 torch.bfloat16
k = k.to(torch.float16)
v = v.to(torch.float16)

# 注意: LSE (log-sum-exp) 始终是 FP32 (数值稳定性)
```

#### GQA/MQA 减少 KV cache

```python
# 标准 MHA: KV cache = [batch, max_len, 32 heads, 128 dim]
# 内存: batch * max_len * 32 * 128 * 2 bytes (FP16)

# GQA (8 KV heads): KV cache = [batch, max_len, 8 heads, 128 dim]
# 内存: batch * max_len * 8 * 128 * 2 bytes
# 节省: 75% !

# MQA (1 KV head): 节省 96.875%
```

#### 释放不需要的张量

```python
# Prefill 后立即删除不需要的张量
del q_prefill, k_prefill, v_prefill
torch.cuda.empty_cache()

# 只保留缓存
```

### 2. 计算优化

#### 使用 torch.compile (PyTorch >= 2.0)

```python
import torch

# 编译注意力函数
compiled_attn = torch.compile(flash_attn_with_kvcache)

# 首次调用会编译 (慢),后续调用快
output = compiled_attn(q_new, cache_k, cache_v, k_new, v_new, ...)
```

#### 融合操作

```python
# 不好: 多次内核调用
q = q_proj(hidden)
q = q / math.sqrt(head_dim)  # 单独的缩放

# 好: 使用 softmax_scale 参数融合
q = q_proj(hidden)
output = flash_attn_func(q, k, v, softmax_scale=1.0/math.sqrt(head_dim))
```

### 3. 批处理优化

#### Continuous Batching

动态调整批次大小,最大化 GPU 利用率:

```python
class ContinuousBatcher:
    def __init__(self, max_batch_size, max_seq_len):
        self.max_batch_size = max_batch_size
        self.running_requests = []

    def add_request(self, prompt):
        """添加新请求到批次"""
        if len(self.running_requests) < self.max_batch_size:
            self.running_requests.append({
                'tokens': encode(prompt),
                'cache_k': None,
                'cache_v': None,
                'seqlen': len(encode(prompt))
            })

    def step(self):
        """执行一步生成"""
        # 收集所有请求的 q_new
        batch_q = []
        for req in self.running_requests:
            # ... 计算 q_new ...
            batch_q.append(q_new)

        # 批量推理
        batch_q = torch.cat(batch_q, dim=0)
        # ... 调用 flash_attn_varlen_func ...

        # 检查完成的请求并移除
        self.running_requests = [r for r in self.running_requests
                                 if not r['finished']]

        # 可以添加新请求
        if len(self.running_requests) < self.max_batch_size:
            # 从队列添加等待的请求
            pass
```

### 4. 架构特定优化

#### Ampere (A100)
- 块大小: `BLOCK_M=128, BLOCK_N=64`
- 使用 FP16 (BF16 支持有限)

#### Hopper (H100)
- 块大小: `BLOCK_M=128, BLOCK_N=128` (更大)
- 优先使用 BF16 (更好的数值稳定性)
- 使用 FlashAttention-3 (cute 实现)

#### Blackwell (B100/B200)
- 使用最新的 FA3 实现
- 支持更大的 head_dim (e.g., 192)

---

## 常见模式 / Common Patterns

### 1. 文本生成 (GPT 风格)

```python
def generate(
    prompt: str,
    model: nn.Module,
    max_new_tokens: int = 100,
    temperature: float = 1.0,
    top_k: int = 50
):
    """自回归文本生成"""
    # 1. Prefill
    tokens = tokenizer.encode(prompt)
    hidden = model.embed(tokens)
    q, k, v = model.qkv_proj(hidden)
    cache_k, cache_v = k, v

    attn_out = flash_attn_func(q, k, v, causal=True)
    hidden = model.output_proj(attn_out)
    logits = model.lm_head(hidden[:, -1, :])

    # 2. Decode
    for _ in range(max_new_tokens):
        # 采样
        next_token = sample_top_k(logits, k=top_k, temperature=temperature)
        tokens.append(next_token)

        # 计算新 token
        hidden = model.embed(next_token.unsqueeze(0))
        q_new, k_new, v_new = model.qkv_proj(hidden)

        # 注意力
        attn_out = flash_attn_with_kvcache(
            q_new, cache_k, cache_v, k_new, v_new,
            cache_seqlens=len(tokens) - 1,
            causal=True
        )

        # 下一个 token 的 logits
        hidden = model.output_proj(attn_out)
        logits = model.lm_head(hidden[:, -1, :])

        if next_token == eos_token_id:
            break

    return tokenizer.decode(tokens)
```

### 2. 对话 (ChatGPT 风格)

```python
def chat(messages: List[Dict], model: nn.Module):
    """多轮对话"""
    # 格式化对话历史
    # [{"role": "user", "content": "..."}, {"role": "assistant", "content": "..."}, ...]
    prompt = format_chat_prompt(messages)

    # 生成回复
    response = generate(prompt, model, max_new_tokens=512)

    return response
```

### 3. 前缀缓存 (Prefix Caching)

对于重复的提示词前缀,缓存 KV:

```python
# 系统提示 (所有请求共享)
system_prompt = "You are a helpful assistant."
system_tokens = tokenizer.encode(system_prompt)

# 预计算并缓存系统提示的 KV
system_hidden = model.embed(system_tokens)
system_k, system_v = model.kv_proj(system_hidden)

# 对于每个用户请求
def generate_with_prefix(user_input):
    user_tokens = tokenizer.encode(user_input)
    user_hidden = model.embed(user_tokens)
    user_q, user_k, user_v = model.qkv_proj(user_hidden)

    # 组合系统和用户的 KV
    cache_k = torch.cat([system_k, user_k], dim=1)
    cache_v = torch.cat([system_v, user_v], dim=1)

    # Decode
    # ...
```

### 4. 思维链推理 (Chain-of-Thought)

```python
def cot_generate(problem: str, model: nn.Module):
    """思维链推理"""
    # 添加 CoT 提示
    cot_prompt = f"{problem}\nLet's think step by step:\n"

    # 生成推理步骤
    reasoning = generate(cot_prompt, model, max_new_tokens=300)

    # 提取最终答案
    answer_prompt = f"{cot_prompt}{reasoning}\nTherefore, the answer is:"
    answer = generate(answer_prompt, model, max_new_tokens=50)

    return reasoning, answer
```

---

## 调试和分析 / Debugging and Profiling

### 1. 检查缓存正确性

```python
# 验证 KV cache 使用是否正确
# 方法 1: 从头计算 (慢但正确)
all_tokens = prompt_tokens + generated_tokens
full_output = model(all_tokens)

# 方法 2: 使用 KV cache (快)
cached_output = generate_with_cache(prompt_tokens, len(generated_tokens))

# 比较
assert torch.allclose(full_output, cached_output, atol=1e-2)
```

### 2. 性能分析

```python
import torch.profiler

with torch.profiler.profile(
    activities=[torch.profiler.ProfilerActivity.CUDA],
    with_stack=True
) as prof:
    generate(prompt, model, max_new_tokens=100)

print(prof.key_averages().table(sort_by="cuda_time_total"))
```

### 3. 内存分析

```python
# 监控内存使用
print(f"Allocated: {torch.cuda.memory_allocated() / 1e9:.2f} GB")
print(f"Reserved: {torch.cuda.memory_reserved() / 1e9:.2f} GB")

# KV cache 大小估计
kv_cache_size = (
    2 *  # K 和 V
    batch_size *
    max_seq_len *
    num_heads *
    head_dim *
    2  # FP16 = 2 bytes
) / 1e9  # 转换为 GB

print(f"KV cache: {kv_cache_size:.2f} GB")
```

---

## 常见错误和解决方案 / Common Errors and Solutions

### 1. 内存不足 (OOM)

**症状**:
```
RuntimeError: CUDA out of memory
```

**解决方案**:
- 减少 `max_seq_len`
- 使用 GQA/MQA 减少 KV cache
- 使用 Paged KV cache
- 减少 `batch_size`
- 使用 FP16 而非 FP32

### 2. 形状不匹配

**症状**:
```
RuntimeError: shape mismatch
```

**解决方案**:
- 检查 `q.shape[2]` (num_heads) 是否能被 `k.shape[2]` (num_heads_kv) 整除
- 确保 `head_dim` 是 8 的倍数
- 检查 `cache_seqlens` 是否正确

### 3. 数值不稳定

**症状**: 输出包含 NaN 或 Inf

**解决方案**:
- 使用 BF16 而非 FP16 (更好的数值范围)
- 检查 `softmax_scale` 是否合理
- 检查输入是否包含异常值

---

## 最佳实践总结 / Best Practices Summary

✅ **推荐做法**:
1. 预分配 KV cache (避免动态增长)
2. 使用 GQA/MQA 减少内存
3. 使用 varlen 批处理 (避免 padding)
4. 利用 RoPE 内核融合
5. 使用 `torch.no_grad()` (推理无需梯度)
6. 监控 GPU 利用率和内存

❌ **避免做法**:
1. 每步重新计算所有 K/V
2. 使用过大的 `max_seq_len` (浪费内存)
3. 混合 FP32 和 FP16 (不必要的转换)
4. 忽略批处理 (GPU 利用率低)

---

## 参考资料 / References

- [vLLM](https://github.com/vllm-project/vllm): 高性能 LLM 推理引擎
- [PagedAttention 论文](https://arxiv.org/abs/2309.06180)
- [Continuous Batching 博客](https://www.anyscale.com/blog/continuous-batching-llm-inference)

---

## 附录: 完整推理示例 / Appendix: Complete Inference Example

```python
# 完整的 LLM 推理示例 (简化版)
import torch
import math
from flash_attn import flash_attn_func, flash_attn_with_kvcache

class SimpleLLM:
    def __init__(self, config):
        self.config = config
        # ... 加载模型权重 ...

    def generate(
        self,
        prompt: str,
        max_new_tokens: int = 100,
        temperature: float = 1.0
    ):
        # 1. Prefill
        tokens = self.tokenizer.encode(prompt)
        B, L = 1, len(tokens)
        H, D = self.config.num_heads, self.config.head_dim
        H_kv = self.config.num_heads_kv

        # 预分配 cache
        max_len = L + max_new_tokens
        cache_k = torch.zeros(B, max_len, H_kv, D, dtype=torch.float16, device='cuda')
        cache_v = torch.zeros(B, max_len, H_kv, D, dtype=torch.float16, device='cuda')

        with torch.no_grad():
            # Embed + position
            x = self.embed(tokens)  # [1, L, hidden_dim]

            # 所有层
            for layer in self.layers:
                # 1. Attention
                q = layer.q_proj(x)  # [1, L, H, D]
                k = layer.k_proj(x)  # [1, L, H_kv, D]
                v = layer.v_proj(x)  # [1, L, H_kv, D]

                attn_out = flash_attn_func(
                    q, k, v,
                    causal=True,
                    softmax_scale=1.0 / math.sqrt(D)
                )  # [1, L, H, D]

                # 存储 K/V 到 cache
                cache_k[:, :L, :, :] = k
                cache_v[:, :L, :, :] = v

                # 2. FFN
                x = layer.ffn(x + attn_out)

            # Logits
            logits = self.lm_head(x[:, -1, :])  # [1, vocab_size]

        # 2. Decode
        cache_seqlen = L
        for step in range(max_new_tokens):
            # Sample
            next_token = self.sample(logits, temperature)
            tokens.append(next_token)

            if next_token == self.eos_token_id:
                break

            with torch.no_grad():
                x = self.embed([next_token])  # [1, 1, hidden_dim]

                for layer_idx, layer in enumerate(self.layers):
                    q = layer.q_proj(x)  # [1, 1, H, D]
                    k = layer.k_proj(x)  # [1, 1, H_kv, D]
                    v = layer.v_proj(x)  # [1, 1, H_kv, D]

                    attn_out = flash_attn_with_kvcache(
                        q=q,
                        k_cache=cache_k,
                        v_cache=cache_v,
                        k=k,
                        v=v,
                        cache_seqlens=cache_seqlen,
                        causal=True,
                        softmax_scale=1.0 / math.sqrt(D)
                    )  # [1, 1, H, D]

                    x = layer.ffn(x + attn_out)

                logits = self.lm_head(x[:, -1, :])

            cache_seqlen += 1

        return self.tokenizer.decode(tokens)
```

---

本文档持续更新中...
