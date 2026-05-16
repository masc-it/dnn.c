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

/* ── Scaled Dot-Product Attention (Causal) ──
 *
 *   output = causal_softmax(Q @ K^T / sqrt(d_k) + mask) @ V
 *
 *   Q, K, V — input tensors. Shape: [B, H, N, d_head] (4D contiguous only).
 *   Non-4D callers must unsqueeze.  All inputs MUST share ndim and d_head.
 *   Inputs MUST be contiguous.
 *
 *   mask — additive attention mask broadcastable to [B, H, N, N], or NULL.
 *          Causal mask is always applied implicitly.
 *
 *   Returns tensor with shape [B, H, N, d_head].
 *
 *   Internally uses row-blocked triangular matmuls (tile size = DNN_ATTENTION_TILE_ROWS)
 *   to avoid computing future-token attention scores.  See docs/optim/tri_attn.md.
 */
tensor *tensor_attention(struct mem_pool *scratch,
                         tensor *Q, tensor *K, tensor *V,
                         tensor *mask);

#endif /* DNN_ATTENTION_H */
