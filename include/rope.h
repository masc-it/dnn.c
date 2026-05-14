#ifndef DNN_ROPE_H
#define DNN_ROPE_H

#include "tensor.h"

/* ── RoPE frequency table ──
 *
 * Compute the frequency table for Rotary Position Embedding.
 *
 *   theta_k = base^{-2k/d}   for k = 0 .. d/2 - 1
 *
 * Returns a 1D contiguous tensor of length d/2 allocated from the
 * scratch pool, with requires_grad=0.  This is a constant table —
 * no grad_fn attached.
 *
 * Parameters:
 *   d    — head dimension (must be even, > 0)
 *   base — base frequency (10000.0 in the original RoPE paper)
 */
tensor *tensor_rope_freqs(int d, float base);

#endif /* DNN_ROPE_H */
