#ifndef DNN_TRANSFORMER_H
#define DNN_TRANSFORMER_H

#include "tensor.h"
#include "module.h"
#include "nn.h"
#include "attention.h"   /* attention_mode */

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
kv_cache *kv_cache_create(struct mem_pool *params_pool, int B, int H, int max_seq, int d_k);

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
tensor *kv_cache_get_K(struct mem_pool *scratch, kv_cache *kvc);

/* Get view of valid V portion [B, H, seq_len, d_k] (slice along dim 2). */
tensor *kv_cache_get_V(struct mem_pool *scratch, kv_cache *kvc);

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
    module    base;                 /* first field */
    linear    *qkv_proj;            /* d_model → 3 * n_heads * d_k (fused QKV) */
    linear    *out_proj;            /* n_heads * d_k → d_model */
    int        n_heads;
    int        d_k;
    int        d_model;
    layer_norm *attn_norm;         /* pre-attention layer norm */
    layer_norm *ffn_norm;          /* pre-FFN layer norm */
    swiglu_ffn *ffn;               /* d_model → intermediate → d_model */
    /* RoPE frequency tables (borrowed from decoder_lm, not owned) */
    tensor    *freqs_cos;           /* [max_seq_len, d_k/2], NULL = no RoPE */
    tensor    *freqs_sin;           /* [max_seq_len, d_k/2], NULL = no RoPE */
} transformer_block;

transformer_block *transformer_block_create(struct mem_pool *params_pool, int d_model, int n_heads, int d_k,
                                             int intermediate_size);

/* Forward with attention mode selection.
 *
 *   mode       — ATTENTION_CAUSAL or ATTENTION_PREFIX_LM.
 *   prefix_len — bidirectional prefix length (0 for causal).
 *   seq_lens   — nullable [B]; per-batch effective lengths for P1 padding.
 */
tensor *transformer_block_forward_ex(struct mem_pool *scratch,
                                     transformer_block *block,
                                     const tensor *x,
                                     attention_mode mode,
                                     int prefix_len,
                                     const int *seq_lens);

/* Causal forward (backward-compat wrapper). */
tensor *transformer_block_forward(struct mem_pool *scratch,
                                  transformer_block *block,
                                  const tensor *x);

/* ── Cached forward with mode (eval-only, generation) ──
 *
 * Like transformer_block_forward_cached but supports ATTENTION_PREFIX_LM
 * for VLM prefix prefill (must have cache->seq_len == 0).
 */
tensor *transformer_block_forward_cached_ex(struct mem_pool *scratch,
                                            transformer_block *block,
                                            const tensor *x,
                                            kv_cache *cache,
                                            attention_mode mode,
                                            int prefix_len);

/* ── Cached forward (eval-only, generation, causal) ──
 *
 * Forward one token through a single transformer block using KV-cache.
 *
 *   x     — [B, N_new, d_model] tensor for new token(s), MUST be contiguous
 *   cache — KV-cache for this block (stores full K/V history)
 *
 *   Appends new K/V to cache, then does attention using cached K/V.
 *   No autograd wired (generation runs in dnn_no_grad mode).
 *   Returns [B, N_new, d_model].
 */
tensor *transformer_block_forward_cached(struct mem_pool *scratch,
                                          transformer_block *block,
                                          const tensor *x,
                                          kv_cache *cache);

/* ── Parameter count ── */
long long transformer_block_num_parameters(transformer_block *block);

#endif /* DNN_TRANSFORMER_H */
