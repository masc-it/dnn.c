#ifndef DNN_TRANSFORMER_H
#define DNN_TRANSFORMER_H

#include "tensor.h"

/* ── KV Cache ──
 *
 * Pre-allocated K/V buffers for autoregressive generation.
 * Allocates full [B, H, max_seq, d_k] tensors in params pool.
 *
 * Append writes new tokens at position seq_len — copies data from
 * the (contiguous) new K/V tensors directly into the cache buffer
 * at the correct offset.  No scratch allocs during append.
 *
 * Get returns a tensor_slice view of the valid portion [B, H, seq_len, d_k]
 * for use in attention.
 *
 * No autograd — generation is eval-only (dnn_no_grad context).
 * KV-cache is NOT used during training (teacher forcing computes
 * full sequence in one shot).
 */

typedef struct {
    tensor *k_cache;   /* [B, H, max_seq, d_k], params pool */
    tensor *v_cache;   /* [B, H, max_seq, d_k], params pool */
    int     seq_len;   /* current valid length */
    int     max_seq;
} kv_cache;

/* Create KV cache.  Allocates zero-filled [B, H, max_seq, d_k] tensors
 * from the params pool.  seq_len starts at 0. */
kv_cache *kv_cache_create(int B, int H, int max_seq, int d_k);

/* Append new K, V tensors at position seq_len.
 *
 *   K_new, V_new — shape [B, H, N_new, d_k], MUST be contiguous.
 *   Copies N_new tokens into the cache and advances seq_len.
 *   Asserts that seq_len + N_new <= max_seq.
 *
 *   No autograd wired — eval-only memcpy into cache buffer.
 */
void kv_cache_append(kv_cache *kvc, const tensor *K_new, const tensor *V_new);

/* Get view of valid K portion [B, H, seq_len, d_k] (slice along dim 2).
 * Returns a lightweight tensor_slice view sharing the cache's data buffer. */
tensor *kv_cache_get_K(kv_cache *kvc);

/* Get view of valid V portion [B, H, seq_len, d_k] (slice along dim 2). */
tensor *kv_cache_get_V(kv_cache *kvc);

#endif /* DNN_TRANSFORMER_H */
