#ifndef DNN_MULTIHEAD_H
#define DNN_MULTIHEAD_H

#include "tensor.h"

/* ── Split/merge heads for multi-head attention ──
 *
 * Multi-head attention uses a 4D tensor shape [B, H, N, d_k] for
 * per-head Q, K, V.  The QKV projection linear outputs a 3D tensor
 * [B, N, H * d_k] where heads are concatenated along the last dim.
 *
 * split_heads: [B, N, H * d_k] → [B, H, N, d_k]  (new alloc, contiguous)
 * merge_heads: [B, H, N, d_k] → [B, N, H * d_k]  (new alloc, contiguous)
 *
 * Both allocate fresh contiguous output in scratch pool.
 * Autograd wired — backward flows gradients through the inverse layout transform.
 */

tensor *tensor_split_heads(tensor *t, int H);
tensor *tensor_merge_heads(tensor *t);

#endif /* DNN_MULTIHEAD_H */
