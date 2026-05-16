#ifndef DNN_OPS_H
#define DNN_OPS_H

#include "tensor.h"

/* ── Arithmetic ── */
tensor *tensor_add(struct mem_pool *scratch, const tensor *a, const tensor *b);
tensor *tensor_sub(struct mem_pool *scratch, const tensor *a, const tensor *b);
tensor *tensor_mul(struct mem_pool *scratch, const tensor *a, const tensor *b);  /* element-wise */
tensor *tensor_div(struct mem_pool *scratch, const tensor *a, const tensor *b);

/* ── Matrix ops ── */
tensor *tensor_matmul(struct mem_pool *scratch, const tensor *a, const tensor *b);

/* Fused matmul + bias: out = (a @ b) + bias  (if trans_b==0)
 * or                out = (a @ b^T) + bias (if trans_b!=0).
 *
 * Avoids intermediate tensor allocation vs separate matmul + add.
 * bias must be 1-D with shape equal to the last dim of the matmul output
 * (shape[b->ndim-1] when trans_b==0, shape[b->ndim-2] when trans_b!=0). */
tensor *tensor_matmul_add(struct mem_pool *scratch, const tensor *a,
                          const tensor *b, int trans_b, const tensor *bias);

/* ── Activations ── */
tensor *tensor_relu(struct mem_pool *scratch, const tensor *t);
tensor *tensor_sigmoid(struct mem_pool *scratch, const tensor *t);
tensor *tensor_tanh(struct mem_pool *scratch, const tensor *t);
tensor *tensor_silu(struct mem_pool *scratch, const tensor *t);
tensor *tensor_swiglu(struct mem_pool *scratch, const tensor *gate, const tensor *up);
tensor *tensor_softmax(struct mem_pool *scratch, const tensor *t, int dim);
tensor *tensor_causal_softmax(struct mem_pool *scratch, const tensor *t);

/* ── Attention ── */
tensor *tensor_attention(struct mem_pool *scratch, tensor *q, tensor *k, tensor *v, tensor *mask);

/* ── Regularization ── */
tensor *tensor_dropout(struct mem_pool *scratch, const tensor *t, float p);

/* ── Reduction ── */
tensor *tensor_sum(struct mem_pool *scratch, const tensor *t, int dim);
tensor *tensor_mean(struct mem_pool *scratch, const tensor *t, int dim);

/* ── Loss ── */
tensor *tensor_cross_entropy(struct mem_pool *scratch, const tensor *logits, const tensor *target, int dim);

/* ── Masked cross-entropy loss ──
 *
 *   logits: [B,...,V] float — unnormalized class scores.
 *   target: [B,...]   int   — ground-truth class indices.
 *   mask:   [B,...]   float 0/1 — 1 = contribute to loss, 0 = ignore.
 *   dim:    class dimension in logits (e.g. 2 for [B,T,V]).
 *
 *   loss = sum(mask[b,t] * CE(b,t)) / max(sum(mask), 1)
 *   Backward: dlogits skipped where mask == 0.
 */
tensor *tensor_cross_entropy_masked(struct mem_pool *scratch,
                                    const tensor *logits,
                                    const tensor *target,
                                    const tensor *mask,
                                    int dim);

/* ── Embedding ── */
tensor *tensor_embedding(struct mem_pool *scratch, const tensor *table, const tensor *ids);

/* ── Utility ── */
tensor *tensor_pow(struct mem_pool *scratch, const tensor *t, float exp);
tensor *tensor_neg(struct mem_pool *scratch, const tensor *t);
tensor *tensor_triu(struct mem_pool *scratch, int N, int diagonal);

/* ── Pooling ── */
tensor *tensor_avg_pool2d(struct mem_pool *scratch, const tensor *x, int k, int stride);

/* ── Tensor concatenation ── */
tensor *tensor_cat(struct mem_pool *scratch, const tensor *a, const tensor *b, int dim);

#endif /* DNN_OPS_H */
