#ifndef DNN_VLM_H
#define DNN_VLM_H

#include "tensor.h"
#include "module.h"
#include "nn.h"
#include "gpt.h"
#include "optim.h"
#include "attention.h"

/* ── Vision-Language Model (prefix-LM decoder-only VLM) ──
 *
 * Architecture: image patch embedding → [B,I,D] + text token embeds [B,T,D]
 *   → concat to [B,I+T,D] → decoder-only GPT blocks (prefix-LM attention)
 *   → norm → tied lm_head → logits [B,I+T,vocab_size]
 *
 * Image tokens (prefix) have bidirectional attention among themselves.
 * Text tokens attend all image tokens + causal text tokens.
 *
 * No cross-attention, no modality embedding in this handoff.
 */

typedef struct vision_lm {
    module      base;             /* first field */

    decoder_lm *lm;               /* text decoder; owns token embed, blocks, norm, lm_head */
    conv2d     *patch_embed;      /* image patch projection: C -> d_model */
    rms_norm   *image_norm;       /* RMSNorm after patch embed, before cat with text */

    tensor     *image_pos;        /* nullable [1, n_img_tokens, d_model], learned */

    int use_image_pos;

    int image_channels;
    int image_h;
    int image_w;
    int patch_size;
    int patch_h;
    int patch_w;
    int n_img_tokens;

    int d_model;
    int vocab_size;
} vision_lm;

/* ── Constructor ──
 *
 *   use_image_pos: 1 = allocate and register learned image positional embeddings
 *
 *   Image H,W must be divisible by patch_size.
 *   d_model must equal n_heads * d_k.
 */
vision_lm *vision_lm_create(struct mem_pool *params_pool,
                            int vocab_size, int d_model,
                            int n_layers, int n_heads, int d_k,
                            int intermediate_size,
                            int image_channels, int image_h, int image_w,
                            int patch_size, int use_image_pos);

/* ── Weight initialization ──
 *
 *   - decoder_lm_init_weights on child LM
 *   - patch_embed weight: Xavier-style normal std sqrt(2/(C*P*P + d_model))
 *   - patch_embed bias: zero
 *   - image_pos: normal std=0.02 (if enabled)
 */
void vision_lm_init_weights(vision_lm *vlm);

/* ── RoPE passthrough ──
 *
 *   Enables RoPE on child LM with max_seq_len = n_img_tokens + max_text_len.
 *   Ensures RoPE length covers combined image+text sequence.
 */
void vision_lm_enable_rope(struct mem_pool *params_pool,
                           vision_lm *vlm,
                           int max_text_len,
                           float base);

/* ── Parameter count ── */
long long vision_lm_num_parameters(vision_lm *vlm);

/* ── Image patch embedding forward ──
 *
 *   images: [B, C, H, W] float, contiguous
 *   Returns: [B, n_img_tokens, d_model]
 */
tensor *vision_lm_image_embeds(struct mem_pool *scratch,
                               vision_lm *vlm,
                               const tensor *images);

/* ── Multimodal embedding builder ──
 *
 *   images:  [B, C, H, W]
 *   text_ids: [B, T] int tensor
 *   Returns: [B, I+T, d_model]
 *
 *   Concat order: [image_patches] + [text_embeds]
 */
tensor *vision_lm_build_embeds(struct mem_pool *scratch,
                               vision_lm *vlm,
                               const tensor *images,
                               const tensor *text_ids);

/* ── Full forward (prefix-LM attention) ──
 *
 *   Returns logits [B, I+T, vocab_size]
 */
tensor *vision_lm_forward(struct mem_pool *scratch,
                          vision_lm *vlm,
                          const tensor *images,
                          const tensor *text_ids);

/* ── Text-only logits helper ──
 *
 *   Returns [B, T, vocab_size] logits aligned to text input positions.
 *   logits_text[b,t] is the prediction before target_ids[b,t].
 */
tensor *vision_lm_forward_text_logits(struct mem_pool *scratch,
                                      vision_lm *vlm,
                                      const tensor *images,
                                      const tensor *text_ids);

/* ── Target shift helper ──
 *
 *   text_ids [B, T] → [B, T-1] target (reuses decoder_lm_shift_targets).
 */
tensor *vision_lm_shift_targets(struct mem_pool *pool,
                                const tensor *text_ids);

/* ── Training step (full text CE, fixed-length batches) ──
 *
 *   input_ids:  [B, T]  e.g. [BOS, y0, ..., y_{T-2}]
 *   target_ids: [B, T]  e.g. [y0,  y1, ..., y_{T-1}]
 *
 *   Trains on every text position (no mask).  Use exact-length
 *   buckets for variable-length labels.  Returns scalar loss.
 */
tensor *vision_lm_train_step(struct mem_pool *scratch_pool,
                             vision_lm *vlm,
                             const tensor *images,
                             const tensor *input_ids,
                             const tensor *target_ids,
                             adamw_opt *opt,
                             float grad_clip,
                             float *grad_norm_out);

/* ── Autoregressive generation ──
 *
 *   image:      [1, C, H, W]
 *   prompt_ids: [1, T] int tensor
 *   max_new_tokens > 0
 *   temperature >= 0  (0 = greedy argmax)
 *   use_cache: 1 = KV-cache, 0 = full reforward each step
 *   n_out: receives output length
 *
 *   Returns generated token IDs (prompt + new) from data_pool.
 *   Image prefix is implicit (not in output).
 *   Must run inside no-grad context (handled internally).
 */
int *vision_lm_generate(struct mem_pool *scratch_pool,
                        struct mem_pool *data_pool,
                        vision_lm *vlm,
                        const tensor *image,
                        const tensor *prompt_ids,
                        int max_new_tokens,
                        float temperature,
                        int use_cache,
                        int *n_out);

/* ── Padded loss (variable-length batches with optimized attention) ──
 *
 *   images:     [B, C, H, W]
 *   input_ids:  [B, Tmax]  (padded with TOKENIZER_PAD_ID)
 *   target_ids: [B, Tmax]
 *   loss_mask:  [B, Tmax] float 0/1 — 0 at PAD positions, 1 at real targets
 *   text_lens:  [B] int — valid text length per sample (before padding)
 *
 *   Builds prefix-LM seq_lens and returns masked CE. Caller owns backward,
 *   clipping, optimizer step, and instrumentation.
 */
tensor *vision_lm_loss_padded(struct mem_pool *scratch_pool,
                               vision_lm *vlm,
                               const tensor *images,
                               const tensor *input_ids,
                               const tensor *target_ids,
                               const tensor *loss_mask,
                               const int *text_lens);

/* ── Padded training step (variable-length batches with optimized attention) ──
 *
 *   Convenience wrapper around vision_lm_loss_padded + backward + clip + step.
 */
tensor *vision_lm_train_step_padded(struct mem_pool *scratch_pool,
                                     vision_lm *vlm,
                                     const tensor *images,
                                     const tensor *input_ids,
                                     const tensor *target_ids,
                                     const tensor *loss_mask,
                                     const int *text_lens,
                                     adamw_opt *opt,
                                     float grad_clip,
                                     float *grad_norm_out);

/* ── Masked training step (instruction tuning / answer-only supervision) ──
 *
 *   Like vision_lm_train_step but uses masked CE so only positions
 *   where loss_mask[b,t]=1 contribute to the loss.
 *   Uses regular (non-padded) attention — all samples same T.
 */
tensor *vision_lm_train_step_masked(struct mem_pool *scratch_pool,
                                     vision_lm *vlm,
                                     const tensor *images,
                                     const tensor *input_ids,
                                     const tensor *target_ids,
                                     const tensor *loss_mask,
                                     adamw_opt *opt,
                                     float grad_clip,
                                     float *grad_norm_out);

#endif /* DNN_VLM_H */
