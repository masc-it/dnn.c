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

tensor *tensor_split_heads(struct mem_pool *scratch, tensor *t, int H);
tensor *tensor_merge_heads(struct mem_pool *scratch, tensor *t);

/* ── Fused QKV split + split heads (for fused QKV projection) ──
 *
 * Takes the fused output of qkv_proj:  [B, N, 3*H*d_k]  (contiguous)
 * Produces 3 split-head outputs:      [B, H, N, d_k]   each (contiguous)
 *
 * Reads contiguous from source (no strided access) and writes contiguous
 * to each output.  Single pass — much faster than slice + 3× split_heads.
 *
 * Autograd wired: backward accumulates gradients from all three outputs
 * back to the single fused qkv tensor.
 */
void tensor_split_qkv_heads(struct mem_pool *scratch,
                            tensor *qkv, int H,
                            tensor **Qh_out, tensor **Kh_out, tensor **Vh_out);

#endif /* DNN_MULTIHEAD_H */
