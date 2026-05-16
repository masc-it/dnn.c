#ifndef DNN_GPT_H
#define DNN_GPT_H

#include "tensor.h"
#include "module.h"
#include "nn.h"
#include "transformer.h"   /* for transformer_block, kv_cache, attention_mode */
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
    rms_norm            *norm;                /* final RMS norm */
    linear              *lm_head;             /* d_model → vocab_size */
    int                  d_model;
    int                  vocab_size;
} decoder_lm;

/* Create decoder-only LM. */
decoder_lm *decoder_lm_create(struct mem_pool *params_pool,
                               int vocab_size, int d_model,
                               int n_layers, int n_heads, int d_k,
                               int intermediate_size);

/* ── Token embedding lookup helpers ──
 *
 *   decoder_lm_token_embeds: input_ids [B,T] → [B,T,d_model]
 */
tensor *decoder_lm_token_embeds(struct mem_pool *scratch,
                                 decoder_lm *lm,
                                 const tensor *input_ids);

/* ── Forward from already-built embeddings [B,S,d_model] ──
 *
 *   Runs lm->blocks → lm->norm → lm_head.  Returns logits [B,S,vocab_size].
 */
tensor *decoder_lm_forward_embeds(struct mem_pool *scratch,
                                   decoder_lm *lm,
                                   const tensor *embeds);

/* Extended version: supports attention mode, prefix_len, and per-batch seq_lens. */
tensor *decoder_lm_forward_embeds_ex(struct mem_pool *scratch,
                                      decoder_lm *lm,
                                      const tensor *embeds,
                                      attention_mode mode,
                                      int prefix_len,
                                      const int *seq_lens);

/* ── Hidden forward only: embeds [B,S,D] → final normalized h [B,S,D].
 *   Useful for generation when caller only needs last-token logits. */
tensor *decoder_lm_hidden_from_embeds(struct mem_pool *scratch,
                                       decoder_lm *lm,
                                       const tensor *embeds);

/* Extended version with attention mode. */
tensor *decoder_lm_hidden_from_embeds_ex(struct mem_pool *scratch,
                                          decoder_lm *lm,
                                          const tensor *embeds,
                                          attention_mode mode,
                                          int prefix_len,
                                          const int *seq_lens);

/* ── LM head only: h @ embed->weight^T + lm_head bias ── */
tensor *decoder_lm_lm_head_forward(struct mem_pool *scratch,
                                    decoder_lm *lm,
                                    const tensor *h);

/* ── Forward pass (full) ──
 *
 *   input_ids — [B, N] int tensor (token IDs).
 *   Returns float tensor [B, N, vocab_size] — unnormalized logits.
 */
tensor *decoder_lm_forward(struct mem_pool *scratch,
                            decoder_lm *lm, const tensor *input_ids);

/* ── Shift targets ── */
tensor *decoder_lm_shift_targets(struct mem_pool *pool,
                                  const tensor *input_ids);

/* ── Training step ── */
tensor *decoder_lm_train_step(struct mem_pool *scratch_pool,
                               decoder_lm *lm, const tensor *input_ids,
                               const tensor *target,
                               adamw_opt *opt, float grad_clip,
                               float *grad_norm_out);

/* ── Sampling helpers ── */
int decoder_lm_argmax_token(const float *logits, int vocab_size);
int decoder_lm_sample_with_temp(const float *logits, int vocab_size, float temp);

/* ── Autoregressive generation ── */
int *decoder_lm_generate(struct mem_pool *scratch_pool,
                          struct mem_pool *data_pool,
                          decoder_lm *lm, const tensor *prompt_ids,
                          int max_new_tokens, float temperature,
                          int use_cache, int *n_out);

/* ── RoPE ── */
void decoder_lm_enable_rope(struct mem_pool *params_pool,
                             decoder_lm *lm, int max_seq_len, float base);

/* ── Weight initialization ── */
void decoder_lm_init_weights(decoder_lm *lm);

/* ── Parameter count ── */
long long decoder_lm_num_parameters(decoder_lm *lm);

#endif /* DNN_GPT_H */
