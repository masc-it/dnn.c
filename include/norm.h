#ifndef DNN_NORM_H
#define DNN_NORM_H

#include "tensor.h"

/* ── Layer Normalization ──
 *
 *   y = γ * (x - μ) / √(σ² + ε) + β
 *
 *   where μ, σ² are computed over the last dimension.
 *
 *   x      — input tensor, MUST be contiguous (any ndim, last dim is normalized)
 *   weight — learnable scale, shape [d], may be NULL (treated as 1)
 *   bias   — learnable shift, shape [d], may be NULL (treated as 0)
 *   eps    — small constant for numerical stability
 *
 *   Returns a new tensor with same shape as x.
 */
tensor *tensor_layer_norm(struct mem_pool *scratch, const tensor *x,
                          const tensor *weight, const tensor *bias, float eps);

/* ── RMS Normalization ──
 *
 *   y = x * rsqrt(mean(x²) + ε) * γ
 *
 *   where mean(x²) is computed over the last dimension.
 *   No bias term (vs LayerNorm).  Simpler gradient backward.
 *
 *   x      — input tensor, MUST be contiguous
 *   weight — learnable scale, shape [d], may be NULL (treated as 1)
 *   eps    — small constant for numerical stability
 *
 *   Returns a new tensor with same shape as x.
 */
tensor *tensor_rms_norm(struct mem_pool *scratch, const tensor *x,
                         const tensor *weight, float eps);

#endif /* DNN_NORM_H */
