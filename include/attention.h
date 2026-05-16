#ifndef DNN_ATTENTION_H
#define DNN_ATTENTION_H

#include "tensor.h"

/* ── Tuning constant ──
 * Row-tile size for triangular causal attention.
 * Benchmark 32/64/128 for your hardware.
 */
#ifndef DNN_ATTENTION_TILE_ROWS
#define DNN_ATTENTION_TILE_ROWS 64
#endif

/* ── Attention mode ──
 *
 *   ATTENTION_CAUSAL:    row i attends positions 0..i (standard causal LM).
 *   ATTENTION_PREFIX_LM: first prefix_len rows attend all prefix positions
 *                         (bidirectional image prefix); remaining rows causal.
 */
typedef enum attention_mode {
    ATTENTION_CAUSAL = 0,
    ATTENTION_PREFIX_LM = 1
} attention_mode;

/* ── Scaled Dot-Product Attention with mode (Causal or Prefix-LM) ──
 *
 *   output = softmax(Q @ K^T / sqrt(d_k) + mask) @ V
 *
 *   Q, K, V — input tensors. Shape: [B, H, N, d_head] (4D contiguous only).
 *   Non-4D callers must unsqueeze.  All inputs MUST share ndim and d_head.
 *   Inputs MUST be contiguous.
 *
 *   mask — additive attention mask broadcastable to [B, H, N, N], or NULL.
 *          For ATTENTION_CAUSAL, causal mask is applied implicitly.
 *          For ATTENTION_PREFIX_LM, prefix-LM mask is implicit.
 *
 *   mode       — ATTENTION_CAUSAL or ATTENTION_PREFIX_LM.
 *   prefix_len — number of bidirectional prefix tokens (ignored in CAUSAL mode).
 *   seq_lens   — nullable [B]; per-batch effective combined lengths for P1
 *                padding support. NULL = use full N for all batches.
 *
 *   Returns tensor with shape [B, H, N, d_head].
 */
tensor *tensor_attention_ex(struct mem_pool *scratch,
                            tensor *Q, tensor *K, tensor *V,
                            tensor *mask,
                            attention_mode mode,
                            int prefix_len,
                            const int *seq_lens);

/* ── Scaled Dot-Product Attention (Causal, backward-compat wrapper) ──
 *
 *   Equivalent to tensor_attention_ex(..., ATTENTION_CAUSAL, 0, NULL).
 */
tensor *tensor_attention(struct mem_pool *scratch,
                         tensor *Q, tensor *K, tensor *V,
                         tensor *mask);

#endif /* DNN_ATTENTION_H */
