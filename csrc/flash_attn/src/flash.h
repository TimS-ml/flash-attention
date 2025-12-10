/******************************************************************************
 * Copyright (c) 2023, Tri Dao.
 ******************************************************************************/

#pragma once

#include "namespace_config.h"

#include <cuda.h>
#include <vector>

#include <ATen/cuda/CUDAGeneratorImpl.h> // For at::Generator and at::PhiloxCudaState

namespace FLASH_NAMESPACE {
// Dimension indices for tensor shapes
constexpr int TOTAL_DIM = 0;  // Total sequence length dimension (for variable-length)
constexpr int H_DIM = 1;      // Number of heads dimension
constexpr int D_DIM = 2;      // Head dimension

////////////////////////////////////////////////////////////////////////////////////////////////////

/**
 * Qkv_params: Base parameters for Query, Key, Value tensors
 *
 * This structure holds pointers and strides for the Q, K, V matrices used in attention.
 * Supports flexible memory layouts through configurable strides.
 *
 * Memory Layout:
 * - Batch-first: [batch, seqlen, num_heads, head_dim]
 * - Strides allow different memory layouts and enable zero-copy views
 *
 * Multi-Query Attention (MQA) and Grouped-Query Attention (GQA):
 * - MQA: h_k = 1 (all query heads share one KV head)
 * - GQA: h_k < h (query heads are grouped to share KV heads)
 * - Standard MHA: h_k = h (each query head has its own KV head)
 */
struct Qkv_params {
    using index_t = int64_t;  // Use 64-bit indexing for large tensors

    // Pointers to Q, K, V matrices in device memory
    // Type depends on precision (fp16/bf16), cast at kernel launch
    void *__restrict__ q_ptr;  // Query: [batch, seqlen_q, h, head_dim]
    void *__restrict__ k_ptr;  // Key:   [batch, seqlen_k, h_k, head_dim]
    void *__restrict__ v_ptr;  // Value: [batch, seqlen_k, h_k, head_dim]

    // Strides for accessing elements in memory
    // Stride = number of elements to skip to move to next index in that dimension
    index_t q_batch_stride;  // Stride between batches for Q
    index_t k_batch_stride;  // Stride between batches for K
    index_t v_batch_stride;  // Stride between batches for V
    index_t q_row_stride;    // Stride between sequence positions (rows) for Q
    index_t k_row_stride;    // Stride between sequence positions (rows) for K
    index_t v_row_stride;    // Stride between sequence positions (rows) for V
    index_t q_head_stride;   // Stride between attention heads for Q
    index_t k_head_stride;   // Stride between attention heads for K
    index_t v_head_stride;   // Stride between attention heads for V

    // Number of attention heads
    int h;    // Number of query heads
    int h_k;  // Number of key/value heads (for MQA/GQA)

    // Precomputed ratio for MQA/GQA
    // Each group of (h / h_k) query heads shares one KV head
    int h_h_k_ratio; // h / h_k
};

////////////////////////////////////////////////////////////////////////////////////////////////////

/**
 * Flash_fwd_params: Parameters for FlashAttention forward pass
 *
 * Extends Qkv_params with additional parameters needed for the forward pass,
 * including output buffers, dropout, masking, and various attention variants.
 *
 * Key Features Supported:
 * - Variable-length sequences (via cu_seqlens)
 * - Dropout
 * - Causal masking
 * - Sliding window attention
 * - Rotary position embeddings (RoPE)
 * - KV caching for inference
 * - Paged KV cache
 * - ALiBi position bias
 * - Split-KV (for very long sequences)
 */
struct Flash_fwd_params : public Qkv_params {

    // Output tensor and accumulator (for split-KV)
    void * __restrict__ o_ptr;       // Output: [batch, seqlen_q, h, head_dim]
    void * __restrict__ oaccum_ptr;  // Accumulator for split-KV (if num_splits > 1)

    // Strides for output tensor
    index_t o_batch_stride;
    index_t o_row_stride;
    index_t o_head_stride;

    // Attention probability matrix P (only used if return_softmax=True)
    // P = softmax(QK^T / sqrt(d)) before applying dropout
    void * __restrict__ p_ptr;

    // Log-sum-exp (LSE) for numerical stability and backward pass
    // LSE[i] = log(sum(exp(S[i,:]))) where S = QK^T / sqrt(d)
    // Needed for recomputing attention scores in backward pass
    void * __restrict__ softmax_lse_ptr;      // [batch, h, seqlen_q] or [h, total_q]
    void * __restrict__ softmax_lseaccum_ptr; // Accumulator for split-KV

    // Dimensions
    int b;                 // Batch size
    int seqlen_q;          // Query sequence length
    int seqlen_k;          // Key/Value sequence length
    int seqlen_knew;       // New K/V length to append to cache
    int d;                 // Head dimension
    int seqlen_q_rounded;  // Query length rounded to multiple of tile size
    int seqlen_k_rounded;  // Key length rounded to multiple of tile size
    int d_rounded;         // Head dim rounded to multiple of 8
    int rotary_dim;        // Dimension to apply rotary embeddings (0 = disabled)
    int total_q;           // Total query tokens across batch (for varlen)

    // Softmax scaling factors
    // scale_softmax = 1 / sqrt(head_dim) by default
    float scale_softmax;       // Scaling factor for attention scores
    float scale_softmax_log2;  // scale_softmax * log(2) for numerical reasons

    // Variable-length sequence support (ragged batching)
    // cu_seqlens[i] = cumulative sum of sequence lengths up to batch i
    // Example: cu_seqlens_q = [0, 128, 384, 512] means sequences of length [128, 256, 128]
    int * __restrict__ cu_seqlens_q;  // [batch + 1], cumulative sequence lengths for Q
    int * __restrict__ cu_seqlens_k;  // [batch + 1], cumulative sequence lengths for K/V
    int * __restrict__ leftpad_k;     // [batch], left padding for each K sequence

    // Actual used length of each K sequence (can be less than allocated)
    // Used for KV caching where cache may be pre-allocated larger than needed
    int * __restrict__ seqused_k;

    int *__restrict__ blockmask;  // Block-sparse attention mask (experimental)

    // New K/V tensors to append to KV cache (for incremental decoding)
    void * __restrict__ knew_ptr;  // [batch, seqlen_knew, h_k, d]
    void * __restrict__ vnew_ptr;  // [batch, seqlen_knew, h_k, d]

    // Strides for K_new and V_new
    index_t knew_batch_stride;
    index_t vnew_batch_stride;
    index_t knew_row_stride;
    index_t vnew_row_stride;
    index_t knew_head_stride;
    index_t vnew_head_stride;

    // Rotary Position Embeddings (RoPE)
    // Applied to queries and keys to encode relative positions
    void * __restrict__ rotary_cos_ptr;  // [seqlen, rotary_dim/2]
    void * __restrict__ rotary_sin_ptr;  // [seqlen, rotary_dim/2]

    // Batch indices for KV cache (allows non-contiguous batch indexing)
    int * __restrict__ cache_batch_idx;  // [batch]

    // Paged KV cache (for efficient memory usage in long-context inference)
    // Instead of allocating contiguous KV cache, use a page table to map
    // logical positions to physical memory pages
    int * __restrict__ block_table;        // [batch, max_num_blocks], maps to page indices
    index_t block_table_batch_stride;      // Stride for block table
    int page_block_size;                   // Size of each page (e.g., 256 tokens)

    // Dropout parameters
    float p_dropout;                       // Dropout probability (0.0 = no dropout)
    uint8_t p_dropout_in_uint8_t;          // Dropout threshold as uint8 for fast comparison

    // Dropout scaling
    // Output is scaled by 1/(1-p_dropout) to maintain expected values
    float rp_dropout;                      // Reciprocal of (1 - p_dropout)
    float scale_softmax_rp_dropout;        // scale_softmax * rp_dropout (combined)

    // Sliding window attention (local attention)
    // Query i can only attend to keys in range [i - window_size_left, i + window_size_right]
    int window_size_left;                  // Number of positions to attend backward (-1 = infinite)
    int window_size_right;                 // Number of positions to attend forward (-1 = infinite)

    // Softcapping: cap attention logits to prevent extreme values
    // logits = softcap * tanh(logits / softcap)
    float softcap;                         // Softcap value (0.0 = disabled)

    // Random number generation state for dropout
    at::PhiloxCudaState philox_args;       // Philox RNG state
    uint64_t * rng_state;                  // [2], seed and offset

    // Data type and attention variant flags
    bool is_bf16;                          // True if using bfloat16, false for float16
    bool is_causal;                        // True for causal (autoregressive) masking

    // cu_seqlens_k format flag
    // If true: cu_seqlens_k contains cumulative sums (like cu_seqlens_q)
    // If false: cu_seqlens_k contains actual sequence lengths
    bool is_seqlens_k_cumulative;

    // Rotary embedding format
    // True: interleaved format [x0, x1, ...] -> [cos*x0, cos*x1, ...]
    // False: non-interleaved (GPT-NeoX style)
    bool is_rotary_interleaved;

    // Split-KV parallelization
    // For very long sequences, split computation across sequence dimension
    // num_splits > 1 enables parallel processing of KV blocks
    int num_splits;

    // ALiBi (Attention with Linear Biases) slopes
    // Adds position-dependent bias: -alibi_slope[h] * |i - j|
    void * __restrict__ alibi_slopes_ptr;  // [h] or [batch, h]
    index_t alibi_slopes_batch_stride;

    // Layout flags for varlen (variable-length) mode
    bool unpadded_lse;                     // LSE in [h, total_q] format (not [b, h, seqlen_q])
    bool seqlenq_ngroups_swapped;          // Q transposed for GQA optimization
};

////////////////////////////////////////////////////////////////////////////////////////////////////

/**
 * Flash_bwd_params: Parameters for FlashAttention backward pass
 *
 * Extends Flash_fwd_params with gradient tensors and backward-specific parameters.
 *
 * Backward Pass Strategy:
 * FlashAttention saves memory by NOT storing the full attention matrix during forward pass.
 * Instead, it stores only the log-sum-exp (LSE) values and recomputes attention scores
 * during backward pass. This reduces memory from O(batch * heads * N²) to O(batch * heads * N).
 *
 * The backward pass:
 * 1. Recomputes attention scores S = QK^T using saved Q, K, V
 * 2. Recomputes softmax probabilities using saved LSE
 * 3. Computes gradients dQ, dK, dV using dO (gradient of output)
 * 4. Uses tiling strategy similar to forward pass for memory efficiency
 */
struct Flash_bwd_params : public Flash_fwd_params {

    // Gradient tensors (inputs to backward pass)
    void *__restrict__ do_ptr;  // Gradient of output: [batch, seqlen_q, h, d]
    void *__restrict__ dq_ptr;  // Gradient of query: [batch, seqlen_q, h, d]
    void *__restrict__ dk_ptr;  // Gradient of key: [batch, seqlen_k, h_k, d]
    void *__restrict__ dv_ptr;  // Gradient of value: [batch, seqlen_k, h_k, d]

    // Gradient accumulators (for parallel reduction)
    // When splitting work across thread blocks, gradients are accumulated here
    void *__restrict__ dq_accum_ptr;  // Accumulator for dQ
    void *__restrict__ dk_accum_ptr;  // Accumulator for dK
    void *__restrict__ dv_accum_ptr;  // Accumulator for dV

    // Strides for gradient tensors
    index_t do_batch_stride;
    index_t do_row_stride;
    index_t do_head_stride;
    index_t dq_batch_stride;
    index_t dk_batch_stride;
    index_t dv_batch_stride;
    index_t dq_row_stride;
    index_t dk_row_stride;
    index_t dv_row_stride;
    index_t dq_head_stride;
    index_t dk_head_stride;
    index_t dv_head_stride;

    // Softmax gradient sum: sum(dO * O, dim=-1)
    // Used in backward softmax computation
    void *__restrict__ dsoftmax_sum;  // [batch, h, seqlen_q]

    // Deterministic mode: use atomic adds instead of non-deterministic parallel reduction
    // Slower but ensures bitwise reproducible results across runs
    bool deterministic;

    // Stride for split accumulation (when parallelizing across query sequence)
    index_t dq_accum_split_stride;
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template<typename T, int Headdim, bool Is_causal> void run_mha_fwd_(Flash_fwd_params &params, cudaStream_t stream);
template<typename T, int Headdim, bool Is_causal> void run_mha_fwd_splitkv_dispatch(Flash_fwd_params &params, cudaStream_t stream);

template<typename T, int Headdim, bool Is_causal> void run_mha_bwd_(Flash_bwd_params &params, cudaStream_t stream);

}  // namespace FLASH_NAMESPACE
