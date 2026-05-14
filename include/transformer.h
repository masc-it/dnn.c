#ifndef DNN_TRANSFORMER_H
#define DNN_TRANSFORMER_H

#include "tensor.h"
#include "nn.h"

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

/* ── Transformer Block (pre-norm) ──
 *
 * Standard decoder-only transformer block with pre-norm:
 *
 *   x = x + out_proj( merge_heads( causal_attn( split_heads( QKV( norm(x) ) ) ) ) )
 *   x = x + swiglu_ffn( norm(x) )
 *
 * Layer norm before each sublayer, residual connections around each.
 * All parameters allocated in params pool.  Autograd wired through
 * all sub-operations.
 */

typedef struct {
    linear    *q_proj;              /* d_model → n_heads * d_k */
    linear    *k_proj;              /* d_model → n_heads * d_k */
    linear    *v_proj;              /* d_model → n_heads * d_k */
    linear    *out_proj;            /* n_heads * d_k → d_model */
    int        n_heads;
    int        d_k;
    int        d_model;
    tensor    *attn_norm_weight;    /* [d_model], learnable, init 1 */
    tensor    *attn_norm_bias;      /* [d_model], learnable, init 0 */
    tensor    *ffn_norm_weight;     /* [d_model], learnable, init 1 */
    tensor    *ffn_norm_bias;       /* [d_model], learnable, init 0 */
    swiglu_ffn *ffn;               /* d_model → intermediate → d_model */
} transformer_block;

transformer_block *transformer_block_create(int d_model, int n_heads, int d_k,
                                             int intermediate_size);
tensor            *transformer_block_forward(transformer_block *block,
                                              const tensor *x);

#endif /* DNN_TRANSFORMER_H */
