#ifndef DNN_TRANSFORMER_H
#define DNN_TRANSFORMER_H

#include "tensor.h"
#include "module.h"
#include "nn.h"
#include "optim.h"

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
    /* RoPE frequency tables (borrowed from decoder_lm, not owned) */
    tensor    *freqs_cos;           /* [max_seq_len, d_k/2], NULL = no RoPE */
    tensor    *freqs_sin;           /* [max_seq_len, d_k/2], NULL = no RoPE */
} transformer_block;

transformer_block *transformer_block_create(struct mem_pool *params_pool, int d_model, int n_heads, int d_k,
                                             int intermediate_size);
tensor            *transformer_block_forward(struct mem_pool *scratch,
                                              transformer_block *block,
                                              const tensor *x);

/* ── Decoder-only Language Model ──
 *
 * Full autoregressive LM: embed → N×transformer_block → norm → lm_head
 *
 *   input_ids [B, N] (int tokens) → logits [B, N, vocab_size]
 *
 * All parameters in params pool.  Autograd wired through entire graph.
 * No KV-cache during training (teacher forcing computes full seq in one shot).
 */

typedef struct {
    module               base;                /* first field */
    tensor             *embedding_table;      /* [vocab_size, d_model] */
    transformer_block **blocks;               /* [n_layers] */
    int                  n_layers;
    tensor              *norm_weight;         /* [d_model], final layer norm, init 1 */
    tensor              *norm_bias;           /* [d_model], final layer norm, init 0 */
    linear              *lm_head;             /* d_model → vocab_size */
    int                  d_model;
    int                  vocab_size;
} decoder_lm;

/* Create decoder-only LM.
 *
 *   vocab_size       — number of tokens in vocabulary
 *   d_model          — hidden dimension
 *   n_layers         — number of transformer blocks
 *   n_heads          — attention heads per block
 *   d_k              — head dimension (must satisfy n_heads * d_k == d_model)
 *   intermediate     — SwiGLU FFN intermediate size
 *
 * Allocates all parameters from params pool.  Embedding table and lm_head
 * are separate tensors (no weight tying by default).
 */
decoder_lm *decoder_lm_create(struct mem_pool *params_pool,
                               int vocab_size, int d_model,
                               int n_layers, int n_heads, int d_k,
                               int intermediate_size);

/* Forward pass.
 *
 *   input_ids — [B, N] int tensor (token IDs).  Data stored as int* in
 *               the float data region (same pattern as cross_entropy target).
 *               MUST be contiguous.  Batch B and sequence length N are
 *               inferred from input shape.
 *
 *   Returns float tensor [B, N, vocab_size] — unnormalized logits.
 *   Autograd wired: backward flows gradients to all LM parameters.
 */
tensor *decoder_lm_forward(struct mem_pool *scratch,
                            decoder_lm *lm, const tensor *input_ids);

/* ── Training step (next-token prediction, teacher forcing) ──
 *
 * Performs one training step:
 *
 *   1. Forward: logits = decoder_lm_forward(lm, input_ids)  [B, N, vocab]
 *   2. Shift: logits[:, :-1, :] predict input_ids[:, 1:]
 *   3. Loss: cross-entropy over vocab dimension
 *   4. Backward: dnn_backward(loss)
 *   5. Update: adamw_step(opt) + adamw_zero_grad(opt)
 *
 *   input_ids — [B, N] int tensor (token IDs).  Must be contiguous.
 *               N >= 2 (need at least 1 target token).
 *
 *   Returns the scalar loss tensor from scratch pool.
 *   Caller must reset scratch/data pools before next call.
 */
tensor *decoder_lm_train_step(struct mem_pool *scratch_pool,
                               struct mem_pool *data_pool,
                               decoder_lm *lm, const tensor *input_ids,
                               adamw_opt *opt, float grad_clip,
                               float *grad_norm_out);

/* ── Cached forward (eval-only, generation) ──
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

/* ── Autoregressive generation ──
 *
 * Generate token IDs autoregressively from a prompt.
 *
 *   lm             — trained decoder LM
 *   prompt_ids     — [1, N] int tensor, prompt tokens (contiguous)
 *   max_new_tokens — maximum number of new tokens to generate
 *   temperature    — sampling temperature (>=0). 0 = argmax (greedy).
 *   use_cache      — if non-zero, use KV-cache for O(1) per-step inference
 *   n_out          — output: number of tokens in result
 *
 * Returns int array allocated from data pool with token IDs.
 * Caller must NOT free (pool owns it, reset data pool to release).
 *
 * Generation stops when:
 *   - EOS token (TOKENIZER_EOS_ID = 258) is generated
 *   - max_new_tokens is reached
 *   - total sequence length would overflow a reasonable max
 */
int *decoder_lm_generate(struct mem_pool *scratch_pool,
                          struct mem_pool *data_pool,
                          decoder_lm *lm, const tensor *prompt_ids,
                          int max_new_tokens, float temperature,
                          int use_cache, int *n_out);

/* ── RoPE position encoding ──
 *
 * Enable RoPE on all blocks.  Initializes frequency tables in params pool
 * and sets block->freqs_cos/sin on every block.
 *
 *   lm          — decoder LM (all blocks get freqs assigned)
 *   max_seq_len — maximum sequence length (covers training + generation)
 *   base        — RoPE base frequency (0.0 = default 10000.0)
 *
 * Can be called before or after training.  Safe to call once.
 */
void decoder_lm_enable_rope(struct mem_pool *params_pool,
                             decoder_lm *lm, int max_seq_len, float base);

/* ── Weight initialization ── 
 *
 * GPT-2 style init for decoder-only transformer:
 *   Embedding table, Q/K/V, FFN gate/up, LM head: Normal(0, 0.02)
 *   Attention out_proj, FFN down_proj (residual branches): Normal(0, 0.02 / sqrt(2*n_layers))
 *   All biases: zero
 *   Layer norm γ/β: unchanged (1/0 from decoder_lm_create)
 *
 * Call after decoder_lm_create(), before or after decoder_lm_enable_rope(). */
void decoder_lm_init_weights(decoder_lm *lm);

/* ── Parameter count ── */
long long transformer_block_num_parameters(transformer_block *block);
long long decoder_lm_num_parameters(decoder_lm *lm);

#endif /* DNN_TRANSFORMER_H */
