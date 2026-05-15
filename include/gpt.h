#ifndef DNN_GPT_H
#define DNN_GPT_H

#include "tensor.h"
#include "module.h"
#include "nn.h"
#include "transformer.h"   /* for transformer_block, kv_cache */
#include "optim.h"         /* for adamw_opt, clip_grad_norm */

/* ── Decoder-only Language Model (GPT) ──
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
    embedding           *embed;               /* token embedding table */
    transformer_block  **blocks;              /* [n_layers] */
    int                  n_layers;
    layer_norm          *norm;                /* final layer norm */
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
 * Call after decoder_lm_create(), before or after decoder_lm_enable_rope().
 */
void decoder_lm_init_weights(decoder_lm *lm);

/* ── Parameter count ── */
long long decoder_lm_num_parameters(decoder_lm *lm);

#endif /* DNN_GPT_H */
