#ifndef DNN_ROPE_H
#define DNN_ROPE_H

#include "tensor.h"

/* ── Rotary Position Embedding (RoPE) ──
 *
 * Applies pair-wise 2D rotation to the last dimension of x.
 *
 * For pair k (0 <= k < d/2) at position n:
 *   x[..., n, 2k]   = x[..., n, 2k]   * cos[n][k] - x[..., n, 2k+1] * sin[n][k]
 *   x[..., n, 2k+1] = x[..., n, 2k]   * sin[n][k] + x[..., n, 2k+1] * cos[n][k]
 *
 * x MUST be contiguous.  Last dimension d must be even.  Second-to-last
 * dimension is the sequence (position) dimension N.
 *
 * freqs_cos — shape [N, d/2], contiguous.
 * freqs_sin — shape [N, d/2], contiguous.
 *
 * Modifies x's data buffer in-place.  Returns a lightweight view tensor
 * sharing x's buffer, with grad_fn wired for autograd.
 *
 * Backward rotates the incoming gradient by the inverse angles (transpose
 * of the forward rotation matrix), using the same saved cos/sin tables.
 */

tensor *tensor_rope(struct mem_pool *scratch,
                    tensor *x, const tensor *freqs_cos, const tensor *freqs_sin);

/* ── RoPE frequency table initialization ──
 *
 * Computes cos/sin frequency tables for RoPE.
 *
 *   theta_k = base^{-2k/dim}  for k = 0 .. dim/2 - 1
 *   cos[n][k] = cos(n * theta_k)
 *   sin[n][k] = sin(n * theta_k)
 *
 * Both output tensors are allocated from the params pool (persist across
 * training steps).  Shape: [max_seq_len, dim/2].
 *
 * dim must be even (the head dimension of Q/K).
 * base defaults to 10000.0 in Llama/GPT-NeoX; pass 0.0 to use default.
 */

void tensor_rope_freqs_init(struct mem_pool *params,
                            tensor **freqs_cos, tensor **freqs_sin,
                            int dim, int max_seq_len, float base);

#endif /* DNN_ROPE_H */
