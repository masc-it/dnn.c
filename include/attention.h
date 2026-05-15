#ifndef DNN_ATTENTION_H
#define DNN_ATTENTION_H

#include "tensor.h"

/* ── Scaled Dot-Product Attention ──
 *
 *   output = softmax(Q @ K^T / sqrt(d_k) + mask) @ V
 *
 *   Q, K, V — input tensors. Supported shapes:
 *     2D: [N, d_k]          (single sequence, no batch)
 *     3D: [B, N, d_k]       (batched single head)
 *     4D: [B, H, N, d_k]    (batched multi-head)
 *
 *   All inputs MUST have the same ndim and same d_k (last dim).
 *   Inputs MUST be contiguous.
 *
 *   mask — additive attention mask, shape [N, N] or NULL.
 *          Pass the result of tensor_triu(N, 1) for causal masking.
 *          Broadcast over batch/head dims.
 *
 *   Returns tensor with same shape as Q (last dim = d_k).
 */
tensor *tensor_attention(struct mem_pool *scratch,
                         tensor *Q, tensor *K, tensor *V,
                         tensor *mask);

#endif /* DNN_ATTENTION_H */
