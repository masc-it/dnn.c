#include "vlm.h"
#include "gpt.h"
#include "ops.h"
#include "pool.h"
#include "autograd.h"
#include "tokenizer.h"
#include "rope.h"
#include "conv.h"
#include <assert.h>
#include <string.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>

/* ── Box-Muller normal random (duplicated from gpt.c for VLM weight init) ── */

static float _randn(void) {
    float u1 = (float)rand() / (float)RAND_MAX;
    float u2 = (float)rand() / (float)RAND_MAX;
    return sqrtf(-2.0f * logf(u1 + 1e-10f)) * cosf(6.283185307179586f * u2);
}

/* ══════════════════════════════════════════════════════════════════
 *  Constructor
 * ══════════════════════════════════════════════════════════════════ */

vision_lm *vision_lm_create(struct mem_pool *params_pool,
                            int vocab_size, int d_model,
                            int n_layers, int n_heads, int d_k,
                            int intermediate_size,
                            int image_channels, int image_h, int image_w,
                            int patch_size, int use_image_pos) {
    assert(vocab_size > 0 && d_model > 0 && n_layers > 0);
    assert(n_heads > 0 && d_k > 0 && intermediate_size > 0);
    assert(d_model == n_heads * d_k && "d_model must equal n_heads * d_k");
    assert(image_h > 0 && image_w > 0 && patch_size > 0);
    assert(image_h % patch_size == 0 && "image_h must be divisible by patch_size");
    assert(image_w % patch_size == 0 && "image_w must be divisible by patch_size");

    vision_lm *vlm = _mem_pool_alloc(params_pool, sizeof(vision_lm), NULL);
    memset(vlm, 0, sizeof(vision_lm));

    module_init(&vlm->base, params_pool, "vision_lm");

    vlm->vocab_size    = vocab_size;
    vlm->d_model       = d_model;
    vlm->image_channels = image_channels;
    vlm->image_h       = image_h;
    vlm->image_w       = image_w;
    vlm->patch_size    = patch_size;
    vlm->patch_h       = image_h / patch_size;
    vlm->patch_w       = image_w / patch_size;
    vlm->n_img_tokens  = vlm->patch_h * vlm->patch_w;
    vlm->use_image_pos = use_image_pos;

    /* Patch embedding: conv2d(C, d_model, kernel=patch, stride=patch, pad=0) */
    vlm->patch_embed = conv2d_create(params_pool,
                                     image_channels,
                                     d_model,
                                     patch_size,
                                     patch_size,
                                     0);
    module_add_child(&vlm->base, "patch_embed", &vlm->patch_embed->base);

    /* RMSNorm after patch embed, stabilizes vision feature scale */
    vlm->image_norm = rms_norm_create(params_pool, d_model, 1e-5f);
    module_add_child(&vlm->base, "image_norm", &vlm->image_norm->base);

    /* Decoder LM */
    vlm->lm = decoder_lm_create(params_pool, vocab_size, d_model,
                                n_layers, n_heads, d_k, intermediate_size);
    module_add_child(&vlm->base, "lm", &vlm->lm->base);

    /* Learned image positional embeddings */
    if (use_image_pos) {
        vlm->image_pos = tensor_zeros(params_pool, 3,
                                      (int[]){1, vlm->n_img_tokens, d_model}, 1);
        module_param(&vlm->base, "image_pos", vlm->image_pos);
    } else {
        vlm->image_pos = NULL;
    }

    return vlm;
}

/* ══════════════════════════════════════════════════════════════════
 *  Weight initialization
 * ══════════════════════════════════════════════════════════════════ */

void vision_lm_init_weights(vision_lm *vlm) {
    assert(vlm);

    /* Init child LM (GPT-2 style) */
    decoder_lm_init_weights(vlm->lm);

    /* Init patch_embed weight: Xavier-style normal */
    {
        int C = vlm->image_channels;
        int P = vlm->patch_size;
        int D = vlm->d_model;
        float std = sqrtf(2.0f / (float)(C * P * P + D));

        tensor *w = vlm->patch_embed->weight;
        int nw = tensor_numel(w);
        float *wd = tensor_data_ptr(w);
        for (int i = 0; i < nw; i++)
            wd[i] = _randn() * std;

        /* Zero bias */
        tensor *b = vlm->patch_embed->bias;
        int nb = tensor_numel(b);
        memset(tensor_data_ptr(b), 0, (size_t)nb * sizeof(float));
    }

    /* Init image_pos: normal std=0.02 */
    if (vlm->image_pos) {
        int np = tensor_numel(vlm->image_pos);
        float *pd = tensor_data_ptr(vlm->image_pos);
        for (int i = 0; i < np; i++)
            pd[i] = _randn() * 0.02f;
    }
}

/* ══════════════════════════════════════════════════════════════════
 *  RoPE passthrough
 * ══════════════════════════════════════════════════════════════════ */

void vision_lm_enable_rope(struct mem_pool *params_pool,
                           vision_lm *vlm,
                           int max_text_len,
                           float base) {
    assert(vlm);
    int max_seq = vlm->n_img_tokens + max_text_len;
    decoder_lm_enable_rope(params_pool, vlm->lm, max_seq, base);
}

/* ══════════════════════════════════════════════════════════════════
 *  Parameter count
 * ══════════════════════════════════════════════════════════════════ */

long long vision_lm_num_parameters(vision_lm *vlm) {
    return module_num_parameters(&vlm->base);
}

/* ══════════════════════════════════════════════════════════════════
 *  Image patch embedding forward
 * ══════════════════════════════════════════════════════════════════ */

tensor *vision_lm_image_embeds(struct mem_pool *scratch,
                               vision_lm *vlm,
                               const tensor *images) {
    assert(vlm && images);
    assert(images->ndim == 4 && "images must be 4D [B, C, H, W]");
    assert(images->shape[1] == vlm->image_channels);
    assert(images->shape[2] == vlm->image_h);
    assert(images->shape[3] == vlm->image_w);
    assert(tensor_is_contiguous(images) && "images must be contiguous");

    int B = images->shape[0];

    /* Conv2d: [B, C, H, W] -> [B, D, Hp, Wp] */
    tensor *x = conv2d_forward(scratch, vlm->patch_embed, images);

    /* Transpose to [B, Hp, Wp, D] then contiguous */
    x = tensor_transpose(scratch, x, 1, 2);  /* [B, D, Hp, Wp] -> [B, Hp, D, Wp] */
    x = tensor_transpose(scratch, x, 2, 3);  /* [B, Hp, D, Wp] -> [B, Hp, Wp, D] */
    x = tensor_contiguous(scratch, x);

    /* Reshape to [B, n_img_tokens, d_model] */
    tensor *img = tensor_reshape(scratch, x, 3,
                                 (int[]){B, vlm->n_img_tokens, vlm->d_model});

    /* Add learned image positional embeddings if enabled */
    if (vlm->image_pos) {
        img = tensor_add(scratch, img, vlm->image_pos);  /* broadcast [1,I,D] + [1,I,D] */
    }

    return img;
}

/* ══════════════════════════════════════════════════════════════════
 *  Multimodal embedding builder
 * ══════════════════════════════════════════════════════════════════ */

tensor *vision_lm_build_embeds(struct mem_pool *scratch,
                               vision_lm *vlm,
                               const tensor *images,
                               const tensor *text_ids) {
    assert(vlm && images && text_ids);
    assert(images->shape[0] == text_ids->shape[0] && "batch mismatch");

    /* Image patch embeds [B, I, D] */
    tensor *img = vision_lm_image_embeds(scratch, vlm, images);

    /* RMSNorm on vision features before concat with text */
    img = rms_norm_forward(scratch, vlm->image_norm, img);

    /* Text token embeds [B, T, D] */
    tensor *txt = decoder_lm_token_embeds(scratch, vlm->lm, text_ids);

    /* Concat along seq dim: [B, I+T, D] */
    return tensor_cat(scratch, img, txt, 1);
}

/* ══════════════════════════════════════════════════════════════════
 *  Full forward
 * ══════════════════════════════════════════════════════════════════ */

tensor *vision_lm_forward(struct mem_pool *scratch,
                          vision_lm *vlm,
                          const tensor *images,
                          const tensor *text_ids) {
    tensor *embeds = vision_lm_build_embeds(scratch, vlm, images, text_ids);

    /* Use prefix-LM mode with image prefix length = n_img_tokens */
    return decoder_lm_forward_embeds_ex(scratch, vlm->lm, embeds,
                                        ATTENTION_PREFIX_LM,
                                        vlm->n_img_tokens,
                                        NULL);
}

/* ══════════════════════════════════════════════════════════════════
 *  Text-only logits helper
 * ══════════════════════════════════════════════════════════════════ */

tensor *vision_lm_forward_text_logits(struct mem_pool *scratch,
                                      vision_lm *vlm,
                                      const tensor *images,
                                      const tensor *text_ids) {
    int T = text_ids->shape[1];
    tensor *logits = vision_lm_forward(scratch, vlm, images, text_ids);  /* [B, I+T, V] */
    return tensor_slice(scratch, logits, 1, vlm->n_img_tokens, T);
}

/* ══════════════════════════════════════════════════════════════════
 *  Target shift
 * ══════════════════════════════════════════════════════════════════ */

tensor *vision_lm_shift_targets(struct mem_pool *pool,
                                const tensor *text_ids) {
    return decoder_lm_shift_targets(pool, text_ids);
}

/* ══════════════════════════════════════════════════════════════════
 *  Training step (full text CE, fixed-length)
 * ══════════════════════════════════════════════════════════════════ */

tensor *vision_lm_train_step(struct mem_pool *scratch_pool,
                             vision_lm *vlm,
                             const tensor *images,
                             const tensor *input_ids,
                             const tensor *target_ids,
                             adamw_opt *opt,
                             float grad_clip,
                             float *grad_norm_out) {
    assert(vlm && images && input_ids && target_ids && opt);
    assert(input_ids->ndim == 2 && "input_ids must be 2D [B, T]");
    assert(target_ids->ndim == 2 && "target_ids must be 2D [B, T]");
    assert(input_ids->shape[0] == target_ids->shape[0] && "batch mismatch");
    assert(input_ids->shape[1] == target_ids->shape[1] && "seq len mismatch");
    assert(images->shape[0] == input_ids->shape[0] && "image/text batch mismatch");
    assert(input_ids->shape[1] >= 1 && "need at least 1 target token");

    adamw_zero_grad(opt);

    /* Forward: text logits [B, T, vocab_size] */
    tensor *logits_text = vision_lm_forward_text_logits(scratch_pool, vlm, images, input_ids);

    /* Loss: cross-entropy over vocab dim */
    tensor *loss = tensor_cross_entropy(scratch_pool, logits_text, target_ids, 2);

    /* Backward */
    dnn_backward(scratch_pool, loss);

    /* Gradient norm */
    float gn = 0.0f;
    if (grad_clip > 0.0f) {
        gn = clip_grad_norm(opt->params, opt->n_params, grad_clip);
    } else {
        gn = grad_norm(opt->params, opt->n_params);
    }
    if (grad_norm_out) *grad_norm_out = gn;

    /* Update */
    adamw_step(opt);

    return loss;
}

/* ══════════════════════════════════════════════════════════════════
 *  Padded training step (variable-length batches with seq_lens)
 * ══════════════════════════════════════════════════════════════════ */

tensor *vision_lm_train_step_padded(struct mem_pool *scratch_pool,
                                     vision_lm *vlm,
                                     const tensor *images,
                                     const tensor *input_ids,
                                     const tensor *target_ids,
                                     const tensor *loss_mask,
                                     const int *text_lens,
                                     adamw_opt *opt,
                                     float grad_clip,
                                     float *grad_norm_out) {
    assert(vlm && images && input_ids && target_ids && loss_mask && text_lens && opt);
    assert(input_ids->ndim == 2 && "input_ids must be 2D [B, Tmax]");
    assert(target_ids->ndim == 2 && "target_ids must be 2D [B, Tmax]");
    assert(loss_mask->ndim == 2 && "loss_mask must be 2D [B, Tmax]");
    assert(images->shape[0] == input_ids->shape[0] && "image/text batch mismatch");
    assert(input_ids->shape[0] == target_ids->shape[0] && "batch mismatch");
    assert(input_ids->shape[1] == target_ids->shape[1] && "seq len mismatch");
    assert(input_ids->shape[1] == loss_mask->shape[1] && "seq len mismatch");
    assert(input_ids->shape[1] >= 1 && "need at least 1 target token");

    int B = input_ids->shape[0];
    int Tmax = input_ids->shape[1];
    int I = vlm->n_img_tokens;

    /* Build per-batch combined sequence lengths for attention */
    int *combined_lens = _mem_pool_alloc(scratch_pool, B * sizeof(int), NULL);
    for (int b = 0; b < B; b++) {
        int tl = text_lens[b];
        if (tl > Tmax) tl = Tmax;
        combined_lens[b] = I + tl;
    }

    adamw_zero_grad(opt);

    /* Build multimodal embeds [B, I+Tmax, D] */
    tensor *embeds = vision_lm_build_embeds(scratch_pool, vlm, images, input_ids);

    /* Forward with prefix-LM attention and per-batch seq_lens */
    tensor *logits = decoder_lm_forward_embeds_ex(scratch_pool, vlm->lm, embeds,
                                                  ATTENTION_PREFIX_LM,
                                                  I,
                                                  combined_lens);

    /* Extract text logits [B, Tmax, vocab_size] */
    tensor *logits_text = tensor_slice(scratch_pool, logits, 1, I, Tmax);

    /* Masked CE — pad positions have loss_mask=0 and contribute nothing */
    tensor *loss = tensor_cross_entropy_masked(scratch_pool, logits_text, target_ids, loss_mask, 2);

    /* Backward */
    dnn_backward(scratch_pool, loss);

    /* Gradient norm */
    float gn = 0.0f;
    if (grad_clip > 0.0f) {
        gn = clip_grad_norm(opt->params, opt->n_params, grad_clip);
    } else {
        gn = grad_norm(opt->params, opt->n_params);
    }
    if (grad_norm_out) *grad_norm_out = gn;

    /* Update */
    adamw_step(opt);

    return loss;
}

/* ══════════════════════════════════════════════════════════════════
 *  Masked training step (instruction tuning / answer-only supervision)
 * ══════════════════════════════════════════════════════════════════ */

tensor *vision_lm_train_step_masked(struct mem_pool *scratch_pool,
                                     vision_lm *vlm,
                                     const tensor *images,
                                     const tensor *input_ids,
                                     const tensor *target_ids,
                                     const tensor *loss_mask,
                                     adamw_opt *opt,
                                     float grad_clip,
                                     float *grad_norm_out) {
    assert(vlm && images && input_ids && target_ids && loss_mask && opt);
    assert(input_ids->ndim == 2 && "input_ids must be 2D [B, T]");
    assert(target_ids->ndim == 2 && "target_ids must be 2D [B, T]");
    assert(loss_mask->ndim == 2 && "loss_mask must be 2D [B, T]");
    assert(input_ids->shape[0] == target_ids->shape[0] && "batch mismatch");
    assert(input_ids->shape[1] == target_ids->shape[1] && "seq len mismatch");
    assert(input_ids->shape[1] == loss_mask->shape[1] && "seq len mismatch");
    assert(images->shape[0] == input_ids->shape[0] && "image/text batch mismatch");
    assert(input_ids->shape[1] >= 1 && "need at least 1 target token");

    adamw_zero_grad(opt);

    /* Forward: same as normal train_step but uses masked CE */
    tensor *logits_text = vision_lm_forward_text_logits(scratch_pool, vlm, images, input_ids);

    /* Masked CE — only loss_mask=1 positions contribute */
    tensor *loss = tensor_cross_entropy_masked(scratch_pool, logits_text, target_ids, loss_mask, 2);

    /* Backward */
    dnn_backward(scratch_pool, loss);

    /* Gradient norm */
    float gn = 0.0f;
    if (grad_clip > 0.0f) {
        gn = clip_grad_norm(opt->params, opt->n_params, grad_clip);
    } else {
        gn = grad_norm(opt->params, opt->n_params);
    }
    if (grad_norm_out) *grad_norm_out = gn;

    /* Update */
    adamw_step(opt);

    return loss;
}

/* ══════════════════════════════════════════════════════════════════
 *  Autoregressive generation
 * ══════════════════════════════════════════════════════════════════ */

int *vision_lm_generate(struct mem_pool *scratch_pool,
                        struct mem_pool *data_pool,
                        vision_lm *vlm,
                        const tensor *image,
                        const tensor *prompt_ids,
                        int max_new_tokens,
                        float temperature,
                        int use_cache,
                        int *n_out) {
    assert(vlm && image && prompt_ids);
    assert(image->ndim == 4 && "image must be 4D [1, C, H, W]");
    assert(image->shape[0] == 1 && "only batch=1 supported");
    assert(image->shape[1] == vlm->image_channels);
    assert(image->shape[2] == vlm->image_h);
    assert(image->shape[3] == vlm->image_w);
    assert(prompt_ids->ndim == 2 && "prompt_ids must be 2D [1, T]");
    assert(prompt_ids->shape[0] == 1 && "only batch=1 supported");
    assert(prompt_ids->contiguous && "prompt_ids must be contiguous");
    assert(max_new_tokens > 0);
    assert(temperature >= 0.0f);

    int prompt_len = prompt_ids->shape[1];
    int max_len = prompt_len + max_new_tokens;
    int vocab_size = vlm->vocab_size;
    int d_model = vlm->d_model;
    int I = vlm->n_img_tokens;
    int B = 1;

    /* Copy image to data_pool so it survives scratch resets */
    int img_n = tensor_numel(image);
    tensor *image_data = tensor_zeros_data(data_pool, image->ndim, image->shape);
    memcpy(tensor_data_ptr(image_data), tensor_data_ptr(image), (size_t)img_n * sizeof(float));

    /* Allocate output buffer from data pool */
    int *output = _mem_pool_alloc(data_pool, (size_t)max_len * sizeof(int), NULL);
    memcpy(output, prompt_ids->data, (size_t)prompt_len * sizeof(int));
    int cur_len = prompt_len;

    /* Enter no-grad mode */
    dnn_grad_ctx no_grad_ctx = dnn_no_grad_enter();

    /* Use data_pool copy for the rest of generation */
    image = image_data;

    if (use_cache && vlm->lm->n_layers > 0) {
        /* ── KV-cache path ── */
        int d_k = vlm->lm->blocks[0]->d_k;
        int H   = vlm->lm->blocks[0]->n_heads;

        size_t cache_mark = mem_pool_mark(data_pool);

        int max_seq = I + max_len;  /* image prefix + prompt + new tokens */
        kv_cache **caches = _mem_pool_alloc(data_pool,
                                            (size_t)vlm->lm->n_layers * sizeof(kv_cache*), NULL);
        for (int i = 0; i < vlm->lm->n_layers; i++) {
            caches[i] = kv_cache_create(data_pool, B, H, max_seq, d_k);
        }

        /* ── Prefix prefill: image + prompt using prefix-LM attention ── */

        /* Build image embeds [1, I, D] */
        tensor *img = vision_lm_image_embeds(scratch_pool, vlm, image);

        /* Build text embeds [1, prompt_len, D] */
        tensor *txt = decoder_lm_token_embeds(scratch_pool, vlm->lm, prompt_ids);

        /* Concat: [1, I + prompt_len, D] */
        tensor *h = tensor_cat(scratch_pool, img, txt, 1);

        /* Pass through all blocks with prefix-LM prefill */
        for (int i = 0; i < vlm->lm->n_layers; i++) {
            h = transformer_block_forward_cached_ex(scratch_pool, vlm->lm->blocks[i], h,
                                                    caches[i],
                                                    ATTENTION_PREFIX_LM,
                                                    I);
        }

        /* Final norm + lm_head */
        h = rms_norm_forward(scratch_pool, vlm->lm->norm, h);
        tensor *logits = decoder_lm_lm_head_forward(scratch_pool, vlm->lm, h);
        float *ld = tensor_data_ptr(logits);

        /* Sample from last text position: position I + prompt_len - 1 */
        float *last_logit_buf = _mem_pool_alloc(data_pool, (size_t)vocab_size * sizeof(float), NULL);
        memcpy(last_logit_buf, ld + (I + prompt_len - 1) * vocab_size,
               (size_t)vocab_size * sizeof(float));
        mem_pool_reset(scratch_pool);

        int next_id;
        if (temperature == 0.0f) {
            next_id = decoder_lm_argmax_token(last_logit_buf, vocab_size);
        } else {
            next_id = decoder_lm_sample_with_temp(last_logit_buf, vocab_size, temperature);
        }
        output[cur_len++] = next_id;

        if (next_id != TOKENIZER_EOS_ID) {
            /* ── Per-token generation (causal, one token at a time) ── */
            tensor *single_id_data = tensor_zeros_data(data_pool, 1, (int[]){1});

            while (cur_len < max_len) {
                ((int*)single_id_data->data)[0] = output[cur_len - 1];

                tensor *flat_emb = embedding_forward(scratch_pool, NULL, vlm->lm->embed,
                                                     single_id_data);
                tensor *h_token = tensor_reshape(scratch_pool, flat_emb, 3,
                                                 (int[]){1, 1, d_model});

                for (int i = 0; i < vlm->lm->n_layers; i++) {
                    h_token = transformer_block_forward_cached(scratch_pool,
                                                               vlm->lm->blocks[i],
                                                               h_token, caches[i]);
                }

                h_token = rms_norm_forward(scratch_pool, vlm->lm->norm, h_token);
                tensor *logits_t = decoder_lm_lm_head_forward(scratch_pool, vlm->lm, h_token);
                float *ld_t = tensor_data_ptr(logits_t);

                float *last_buf = _mem_pool_alloc(data_pool, (size_t)vocab_size * sizeof(float), NULL);
                memcpy(last_buf, ld_t, (size_t)vocab_size * sizeof(float));
                mem_pool_reset(scratch_pool);

                if (temperature == 0.0f) {
                    next_id = decoder_lm_argmax_token(last_buf, vocab_size);
                } else {
                    next_id = decoder_lm_sample_with_temp(last_buf, vocab_size, temperature);
                }

                output[cur_len++] = next_id;
                if (next_id == TOKENIZER_EOS_ID) break;
            }
        }

        /* Release KV-caches */
        mem_pool_release(data_pool, cache_mark);

    } else {
        /* ── No-cache path: full forward each step ── */
        while (cur_len < max_len) {
            /* Build tensor from current output buffer [1, cur_len] */
            tensor *ids_tensor = tensor_zeros_data(data_pool, 2, (int[]){1, cur_len});
            memcpy(ids_tensor->data, output, (size_t)cur_len * sizeof(int));

            /* Full VLM forward */
            tensor *logits = vision_lm_forward(scratch_pool, vlm, image, ids_tensor);

            /* Last text logit position: I + cur_len - 1 */
            float *ld = tensor_data_ptr(logits);
            float *last_logits = ld + (I + cur_len - 1) * vocab_size;
            float *last_logit_buf = _mem_pool_alloc(data_pool, (size_t)vocab_size * sizeof(float), NULL);
            memcpy(last_logit_buf, last_logits, (size_t)vocab_size * sizeof(float));
            mem_pool_reset(scratch_pool);

            int next_id;
            if (temperature == 0.0f) {
                next_id = decoder_lm_argmax_token(last_logit_buf, vocab_size);
            } else {
                next_id = decoder_lm_sample_with_temp(last_logit_buf, vocab_size, temperature);
            }

            output[cur_len++] = next_id;
            if (next_id == TOKENIZER_EOS_ID) break;
        }
    }

    dnn_no_grad_exit(no_grad_ctx);

    *n_out = cur_len;
    return output;
}
