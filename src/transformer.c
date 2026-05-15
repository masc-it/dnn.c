#include "transformer.h"
#include "nn.h"
#include "norm.h"
#include "ops.h"
#include "attention.h"
#include "multihead.h"
#include "rope.h"
#include "pool.h"
#include "pool_int.h"
#include "tensor_int.h"
#include "autograd.h"
#include "optim.h"
#include "simd.h"
#include <assert.h>
#include <string.h>
#include <math.h>
#include <stdlib.h>

/* BLAS for fast matmul in cached attention */
#if defined(__APPLE__)
#  include <Accelerate/Accelerate.h>
#elif defined(HAVE_CBLAS)
#  include <cblas.h>
#else
#  define NO_CBLAS 1
#endif

/* ── KV Cache ── */

kv_cache *kv_cache_create(int B, int H, int max_seq, int d_k) {
    kv_cache *kvc = mem_params_alloc(sizeof(kv_cache), NULL);
    int shape[4] = {B, H, max_seq, d_k};
    kvc->k_cache = tensor_zeros(4, shape, 0);
    kvc->v_cache = tensor_zeros(4, shape, 0);
    kvc->seq_len = 0;
    kvc->max_seq = max_seq;
    return kvc;
}

void kv_cache_append(kv_cache *kvc, const tensor *K_new, const tensor *V_new) {
    assert(kvc && "kv_cache_append: NULL cache");
    assert(K_new && V_new && "kv_cache_append: NULL tensor");
    assert(K_new->ndim == 4 && V_new->ndim == 4
           && "kv_cache_append: inputs must be 4D [B, H, N, d_k]");

    int B     = K_new->shape[0];
    int H     = K_new->shape[1];
    int N_new = K_new->shape[2];
    int D     = K_new->shape[3];

    assert(B == kvc->k_cache->shape[0] && "kv_cache_append: batch mismatch");
    assert(H == kvc->k_cache->shape[1] && "kv_cache_append: heads mismatch");
    assert(D == kvc->k_cache->shape[3] && "kv_cache_append: d_k mismatch");
    assert(N_new == V_new->shape[2] && "kv_cache_append: K/V seq len mismatch");
    assert(kvc->seq_len + N_new <= kvc->max_seq
           && "kv_cache_append: cache full");

    float *k_data = (float*)kvc->k_cache->data;
    float *v_data = (float*)kvc->v_cache->data;
    const float *k_src = (const float*)K_new->data;
    const float *v_src = (const float*)V_new->data;

    /* Cache strides (pre-allocated with max_seq along dim 2) */
    int cache_stride_h = kvc->k_cache->strides[1];  /* D * max_seq */
    int cache_stride_s = kvc->k_cache->strides[2];  /* D */
    int v_cache_stride_h = kvc->v_cache->strides[1];

    /* Source strides (K_new / V_new are contiguous from split_heads) */
    int src_stride_h = K_new->strides[1];  /* D * N_new if contiguous */
    int src_stride_b = K_new->strides[0];  /* H * D * N_new if contiguous */

    size_t row_bytes = (size_t)N_new * D * sizeof(float);

    for (int b = 0; b < B; b++) {
        for (int h = 0; h < H; h++) {
            /* Destination: cache (b, h, seq_len, 0) */
            int dst_k_off = kvc->k_cache->offset
                          + b * kvc->k_cache->strides[0]
                          + h * cache_stride_h
                          + kvc->seq_len * cache_stride_s;
            int dst_v_off = kvc->v_cache->offset
                          + b * kvc->v_cache->strides[0]
                          + h * v_cache_stride_h
                          + kvc->seq_len * cache_stride_s;

            /* Source: K_new (b, h, 0, 0) */
            int src_k_off = K_new->offset
                          + b * src_stride_b
                          + h * src_stride_h;
            int src_v_off = V_new->offset
                          + b * src_stride_b
                          + h * src_stride_h;

            memcpy(k_data + dst_k_off, k_src + src_k_off, row_bytes);
            memcpy(v_data + dst_v_off, v_src + src_v_off, row_bytes);
        }
    }

    kvc->seq_len += N_new;
}

tensor *kv_cache_get_K(kv_cache *kvc) {
    assert(kvc && "kv_cache_get_K: NULL cache");
    assert(kvc->seq_len > 0 && "kv_cache_get_K: empty cache");
    return tensor_slice(kvc->k_cache, 2, 0, kvc->seq_len);
}

tensor *kv_cache_get_V(kv_cache *kvc) {
    assert(kvc && "kv_cache_get_V: NULL cache");
    assert(kvc->seq_len > 0 && "kv_cache_get_V: empty cache");
    return tensor_slice(kvc->v_cache, 2, 0, kvc->seq_len);
}

/* ── Transformer Block ── */

transformer_block *transformer_block_create(int d_model, int n_heads, int d_k,
                                             int intermediate_size) {
    assert(d_model > 0 && n_heads > 0 && d_k > 0 && intermediate_size > 0);
    assert(d_model == n_heads * d_k && "d_model must equal n_heads * d_k");

    transformer_block *block = mem_params_alloc(sizeof(transformer_block), NULL);
    block->d_model  = d_model;
    block->n_heads  = n_heads;
    block->d_k      = d_k;

    block->q_proj   = linear_create(d_model, n_heads * d_k);
    block->k_proj   = linear_create(d_model, n_heads * d_k);
    block->v_proj   = linear_create(d_model, n_heads * d_k);
    block->out_proj = linear_create(n_heads * d_k, d_model);

    /* Pre-norm params: init γ=1, β=0 */
    block->attn_norm_weight = tensor_zeros(1, (int[]){d_model}, 1);
    block->attn_norm_bias   = tensor_zeros(1, (int[]){d_model}, 1);
    float *wn = tensor_data_ptr(block->attn_norm_weight);
    for (int i = 0; i < d_model; i++) wn[i] = 1.0f;

    block->ffn_norm_weight = tensor_zeros(1, (int[]){d_model}, 1);
    block->ffn_norm_bias   = tensor_zeros(1, (int[]){d_model}, 1);
    float *wf = tensor_data_ptr(block->ffn_norm_weight);
    for (int i = 0; i < d_model; i++) wf[i] = 1.0f;

    block->ffn = swiglu_ffn_create(d_model, intermediate_size);

    /* RoPE disabled by default */
    block->freqs_cos = NULL;
    block->freqs_sin = NULL;

    return block;
}

tensor *transformer_block_forward(transformer_block *block, const tensor *x) {
    assert(block && x);
    assert(x->ndim >= 2);
    assert(x->shape[x->ndim - 1] == block->d_model);

    /* ── Attention sublayer (pre-norm) ── */
    tensor *residual = (tensor*)x;
    tensor *h = tensor_layer_norm(x, block->attn_norm_weight,
                                   block->attn_norm_bias, 1e-5f);

    /* QKV projections: [B, N, d_model] → [B, N, n_heads * d_k] */
    tensor *Q = linear_forward(block->q_proj, h);
    tensor *K = linear_forward(block->k_proj, h);
    tensor *V = linear_forward(block->v_proj, h);

    /* Split heads: [B, N, H*d_k] → [B, H, N, d_k] */
    tensor *Qh = tensor_split_heads(Q, block->n_heads);
    tensor *Kh = tensor_split_heads(K, block->n_heads);
    tensor *Vh = tensor_split_heads(V, block->n_heads);

    /* ── RoPE: apply rotary position encoding to Q, K ── */
    if (block->freqs_cos && block->freqs_sin) {
        /* Slice freq tables to match current sequence length */
        int N = Qh->shape[2];
        tensor *fc = tensor_slice(block->freqs_cos, 0, 0, N);
        tensor *fs = tensor_slice(block->freqs_sin, 0, 0, N);
        Qh = tensor_rope(Qh, fc, fs);
        Kh = tensor_rope(Kh, fc, fs);
    }

    /* Fused causal attention (no extra mask needed) */
    tensor *attn_out = tensor_attention(Qh, Kh, Vh, NULL);

    /* Merge heads: [B, H, N, d_k] → [B, N, H*d_k] */
    tensor *attn_merged = tensor_merge_heads(attn_out);

    /* Output projection: [B, N, H*d_k] → [B, N, d_model] */
    tensor *attn_proj = linear_forward(block->out_proj, attn_merged);

    /* First residual: x = x + attn_proj */
    tensor *x_after_attn = tensor_add(residual, attn_proj);

    /* ── FFN sublayer (pre-norm) ── */
    residual = x_after_attn;
    h = tensor_layer_norm(x_after_attn, block->ffn_norm_weight,
                           block->ffn_norm_bias, 1e-5f);

    tensor *ffn_out = swiglu_ffn_forward(block->ffn, h);

    /* Second residual: return x + ffn_out */
    return tensor_add(residual, ffn_out);
}

/* ── Decoder-only Language Model ── */

decoder_lm *decoder_lm_create(int vocab_size, int d_model,
                               int n_layers, int n_heads, int d_k,
                               int intermediate_size) {
    assert(vocab_size > 0 && d_model > 0 && n_layers > 0);
    assert(n_heads > 0 && d_k > 0 && intermediate_size > 0);
    assert(d_model == n_heads * d_k && "d_model must equal n_heads * d_k");

    decoder_lm *lm = mem_params_alloc(sizeof(decoder_lm), NULL);
    lm->d_model    = d_model;
    lm->vocab_size = vocab_size;
    lm->n_layers   = n_layers;

    /* Embedding table: [vocab_size, d_model], uniform init */
    float bound = 1.0f / sqrtf((float)d_model);
    lm->embedding_table = tensor_uniform(2, (int[]){vocab_size, d_model}, 1, bound);

    /* Transformer blocks */
    lm->blocks = mem_params_alloc(n_layers * sizeof(transformer_block*), NULL);
    for (int i = 0; i < n_layers; i++) {
        lm->blocks[i] = transformer_block_create(d_model, n_heads, d_k,
                                                   intermediate_size);
    }

    /* Final layer norm: γ=1, β=0 */
    lm->norm_weight = tensor_zeros(1, (int[]){d_model}, 1);
    lm->norm_bias   = tensor_zeros(1, (int[]){d_model}, 1);
    float *wn = tensor_data_ptr(lm->norm_weight);
    for (int i = 0; i < d_model; i++) wn[i] = 1.0f;

    /* LM head: d_model → vocab_size */
    lm->lm_head = linear_create(d_model, vocab_size);

    return lm;
}

tensor *decoder_lm_forward(decoder_lm *lm, const tensor *input_ids) {
    assert(lm && input_ids);
    assert(input_ids->ndim == 2 && "decoder_lm_forward: input_ids must be 2D [B, N]");
    assert(input_ids->contiguous && "decoder_lm_forward: input_ids must be contiguous");

    int B = input_ids->shape[0];
    int N = input_ids->shape[1];
    int D = lm->d_model;

    /* Flatten [B, N] → [B*N] for embedding lookup */
    tensor *flat_ids = tensor_flatten((tensor*)input_ids);  /* view, no data copy */

    /* Embed: [B*N] → [B*N, d_model] */
    tensor *h = tensor_embedding(lm->embedding_table, flat_ids);

    /* Reshape to [B, N, d_model] */
    h = tensor_reshape(h, 3, (int[]){B, N, D});

    /* Pass through all transformer blocks */
    for (int i = 0; i < lm->n_layers; i++) {
        h = transformer_block_forward(lm->blocks[i], h);
    }

    /* Final layer norm */
    h = tensor_layer_norm(h, lm->norm_weight, lm->norm_bias, 1e-5f);

    /* LM head: [B, N, d_model] → [B, N, vocab_size] */
    tensor *logits = linear_forward(lm->lm_head, h);

    return logits;
}

/* ── Training step ── */

tensor *decoder_lm_train_step(decoder_lm *lm, const tensor *input_ids,
                               adamw_opt *opt, float grad_clip,
                               float *grad_norm_out) {
    assert(lm && input_ids && opt);
    assert(input_ids->ndim == 2 && "train_step: input_ids must be 2D [B, N]");
    assert(input_ids->contiguous && "train_step: input_ids must be contiguous");
    assert(input_ids->shape[1] >= 2 && "train_step: need N >= 2 for at least 1 target");

    int B = input_ids->shape[0];
    int N = input_ids->shape[1];

    /* ── Forward ── */
    tensor *logits = decoder_lm_forward(lm, input_ids);  /* [B, N, vocab] */

    /* ── Shift: logits[:, :-1, :] predict input_ids[:, 1:] ── */
    tensor *logits_shifted = tensor_slice(logits, 1, 0, N - 1);  /* [B, N-1, vocab] */

    /* Build target = input_ids[:, 1:], same int-into-float trick */
    tensor *target = tensor_zeros_data(2, (int[]){B, N - 1});
    int *td = (int *)target->data;
    int *id = (int *)input_ids->data;
    for (int b = 0; b < B; b++) {
        for (int n = 1; n < N; n++) {
            td[b * (N - 1) + (n - 1)] = id[b * N + n];
        }
    }

    /* ── Loss: cross-entropy over vocab dim ── */
    tensor *loss = tensor_cross_entropy(logits_shifted, target, 2);

    /* ── Backward ── */
    dnn_backward(loss);

    /* ── Gradient clipping ── */
    float gn = 0.0f;
    if (grad_clip > 0.0f) {
        gn = clip_grad_norm(opt->params, opt->n_params, grad_clip);
    } else {
        /* Compute norm even without clipping, for logging */
        double sum_sq = 0.0;
        for (int i = 0; i < opt->n_params; i++) {
            float *g = tensor_grad(opt->params[i]);
            if (!g) continue;
            int n = tensor_numel(opt->params[i]);
            for (int j = 0; j < n; j++)
                sum_sq += (double)g[j] * (double)g[j];
        }
        gn = sqrtf((float)sum_sq);
    }
    if (grad_norm_out) *grad_norm_out = gn;

    /* ── Update ── */
    adamw_step(opt);
    adamw_zero_grad(opt);

    return loss;
}

/* ── Cached transformer block forward (eval-only, generation) ── */

tensor *transformer_block_forward_cached(transformer_block *block,
                                          const tensor *x,
                                          kv_cache *cache) {
    assert(block && x && cache);
    assert(x->contiguous && "cached forward: x must be contiguous");
    assert(x->ndim >= 2);
    assert(x->shape[x->ndim - 1] == block->d_model);

    int B     = x->shape[0];
    int N_new = x->shape[x->ndim - 2];  /* sequence dim for new tokens */
    int H     = block->n_heads;
    int d_k   = block->d_k;

    /* Pre-norm */
    tensor *h = tensor_layer_norm(x, block->attn_norm_weight,
                                   block->attn_norm_bias, 1e-5f);

    /* QKV projections */
    tensor *Q = linear_forward(block->q_proj, h);  /* [B, N_new, H*d_k] */
    tensor *K = linear_forward(block->k_proj, h);
    tensor *V = linear_forward(block->v_proj, h);

    /* Split heads */
    tensor *Qh = tensor_split_heads(Q, H);  /* [B, H, N_new, d_k] */
    tensor *Kh = tensor_split_heads(K, H);
    tensor *Vh = tensor_split_heads(V, H);

    /* ── RoPE: apply to Q, K with position offset ── */
    if (block->freqs_cos && block->freqs_sin) {
        /* New tokens are at positions [cache->seq_len, cache->seq_len + N_new)
         * Slice the freq tables to start at the current cache position. */
        int pos_offset = cache->seq_len;
        tensor *fc_slice = tensor_slice(block->freqs_cos, 0, pos_offset, N_new);
        tensor *fs_slice = tensor_slice(block->freqs_sin, 0, pos_offset, N_new);
        Qh = tensor_rope(Qh, fc_slice, fs_slice);
        Kh = tensor_rope(Kh, fc_slice, fs_slice);
    }

    /* Append new K/V to cache */
    kv_cache_append(cache, Kh, Vh);

    /* Get full cached K/V */
    tensor *K_full = kv_cache_get_K(cache);  /* [B, H, S, d_k] */
    tensor *V_full = kv_cache_get_V(cache);  /* [B, H, S, d_k] */

    int S = K_full->shape[2];
    float scale = 1.0f / sqrtf((float)d_k);

    /* Output allocation */
    tensor *attn_out = _tensor_scratch_create(4, (int[]){B, H, N_new, d_k}, 0);
    float *od = (float*)attn_out->data;
    float *qd = (float*)Qh->data;
    float *kd = (float*)K_full->data;
    float *vd = (float*)V_full->data;

    /* Temp scores [N_new, S] */
    float *scores = mem_scratch_alloc((size_t)N_new * S * sizeof(float), NULL);

    for (int b = 0; b < B; b++) {
        for (int h = 0; h < H; h++) {
            int bh = b * H + h;

            float *q_slice = qd + bh * N_new * d_k;
            float *k_slice = kd + bh * S * d_k;
            float *v_slice = vd + bh * S * d_k;
            float *o_slice = od + bh * N_new * d_k;

            /* scores = Q @ K^T * scale  [N_new, S] */
#if NO_CBLAS
            for (int i = 0; i < N_new; i++)
                for (int j = 0; j < S; j++) {
                    float sum = 0.0f;
                    for (int kk = 0; kk < d_k; kk++)
                        sum += q_slice[i * d_k + kk] * k_slice[j * d_k + kk];
                    scores[i * S + j] = sum * scale;
                }
#else
            cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans,
                        N_new, S, d_k, scale, q_slice, d_k, k_slice, d_k,
                        0.0f, scores, S);
#endif

            /* Softmax over last dim — no causal mask, all past visible
             * Fused online max + sum_exp + NEON SIMD.
             */
            for (int i = 0; i < N_new; i++) {
                float *row = scores + i * S;

                /* ── Fused pass: online max + sum_exp ── */
                float mx = -INFINITY;
                float se = 0.0f;
#if DNN_HAVE_NEON
                {
                    int j = 0;
                    for (; j + 4 <= S; j += 4) {
                        float32x4_t v = vld1q_f32(row + j);
                        float group_max = vmaxvq_f32(v);
                        if (group_max > mx) {
                            se *= expf(mx - group_max);
                            mx = group_max;
                        }
                        float32x4_t shifted = vsubq_f32(v, vdupq_n_f32(mx));
                        se += vaddvq_f32(simd_expf_f32(shifted));
                    }
                    for (; j < S; j++) {
                        float old_mx = mx;
                        if (row[j] > mx) mx = row[j];
                        if (mx != old_mx) se *= expf(old_mx - mx);
                        se += expf(row[j] - mx);
                    }
                }
#else
                for (int j = 0; j < S; j++) {
                    float old_mx = mx;
                    if (row[j] > mx) mx = row[j];
                    if (mx != old_mx) se *= expf(old_mx - mx);
                    se += expf(row[j] - mx);
                }
#endif

                /* ── Write softmax weights (1 pass) ── */
                float inv_se = 1.0f / se;
#if DNN_HAVE_NEON
                {
                    int j = 0;
                    float32x4_t vmx = vdupq_n_f32(mx);
                    float32x4_t vinv_se = vdupq_n_f32(inv_se);
                    for (; j + 4 <= S; j += 4) {
                        float32x4_t v = vld1q_f32(row + j);
                        float32x4_t exp_v = simd_expf_f32(vsubq_f32(v, vmx));
                        vst1q_f32(row + j, vmulq_f32(exp_v, vinv_se));
                    }
                    for (; j < S; j++)
                        row[j] = expf(row[j] - mx) * inv_se;
                }
#else
                for (int j = 0; j < S; j++)
                    row[j] = expf(row[j] - mx) * inv_se;
#endif
            }

            /* O = P @ V  [N_new, d_k] */
#if NO_CBLAS
            for (int i = 0; i < N_new; i++)
                for (int j = 0; j < d_k; j++) {
                    float sum = 0.0f;
                    for (int kk = 0; kk < S; kk++)
                        sum += scores[i * S + kk] * v_slice[kk * d_k + j];
                    o_slice[i * d_k + j] = sum;
                }
#else
            cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
                        N_new, d_k, S, 1.0f, scores, S, v_slice, d_k,
                        0.0f, o_slice, d_k);
#endif
        }
    }

    /* No autograd — generation is eval-only */

    /* Merge heads: [B, H, N_new, d_k] → [B, N_new, H*d_k] */
    tensor *attn_merged = tensor_merge_heads(attn_out);

    /* Output projection */
    tensor *attn_proj = linear_forward(block->out_proj, attn_merged);

    /* First residual */
    tensor *x_after_attn = tensor_add(x, attn_proj);

    /* ── FFN sublayer (pre-norm) ── */
    h = tensor_layer_norm(x_after_attn, block->ffn_norm_weight,
                           block->ffn_norm_bias, 1e-5f);
    tensor *ffn_out = swiglu_ffn_forward(block->ffn, h);

    /* Second residual */
    return tensor_add(x_after_attn, ffn_out);
}


/* ── Sampling helpers ── */

static int _argmax(const float *logits, int vocab_size) {
    int best = 0;
    float best_val = logits[0];
    for (int i = 1; i < vocab_size; i++) {
        if (logits[i] > best_val) {
            best_val = logits[i];
            best = i;
        }
    }
    return best;
}

static int _sample_with_temp(const float *logits, int vocab_size, float temp) {
    /* NumPy-style categorical sampling:
     *   probs = softmax(logits / temp)
     *   sample = np.random.choice(vocab_size, p=probs)
     */
    float inv_temp = 1.0f / temp;

    /* Find max for numerical stability */
    float mx = -INFINITY;
    for (int i = 0; i < vocab_size; i++) {
        float v = logits[i] * inv_temp;
        if (v > mx) mx = v;
    }

    /* Compute exp and sum */
    float sum = 0.0f;
    for (int i = 0; i < vocab_size; i++) {
        sum += expf(logits[i] * inv_temp - mx);
    }

    /* Draw from cumulative distribution */
    float r = (float)rand() / (float)RAND_MAX;
    float cum = 0.0f;
    for (int i = 0; i < vocab_size; i++) {
        cum += expf(logits[i] * inv_temp - mx) / sum;
        if (r < cum) return i;
    }

    /* Fallback (shouldn't reach here due to floating point) */
    return vocab_size - 1;
}


/* ── Autoregressive generation ── */

int *decoder_lm_generate(decoder_lm *lm, const tensor *prompt_ids,
                          int max_new_tokens, float temperature,
                          int use_cache, int *n_out) {
    assert(lm && prompt_ids);
    assert(prompt_ids->ndim == 2 && "generate: prompt_ids must be 2D [1, N]");
    assert(prompt_ids->shape[0] == 1 && "generate: only batch=1 supported");
    assert(prompt_ids->contiguous && "generate: prompt_ids must be contiguous");
    assert(max_new_tokens > 0);
    assert(temperature >= 0.0f && "generate: temperature must be >= 0");

    int prompt_len = prompt_ids->shape[1];
    int max_len = prompt_len + max_new_tokens;
    int vocab_size = lm->vocab_size;
    int d_model = lm->d_model;
    int B = 1;  /* single-batch generation */

    /* Allocate output buffer from data pool */
    int *output = mem_data_alloc((size_t)max_len * sizeof(int), NULL);
    memcpy(output, prompt_ids->data, (size_t)prompt_len * sizeof(int));
    int cur_len = prompt_len;

    /* Enter no-grad mode for generation */
    dnn_grad_ctx no_grad_ctx = dnn_no_grad_enter();

    if (use_cache && lm->n_layers > 0) {
        /* ── KV-cache path ── */
        int d_k = lm->blocks[0]->d_k;
        int H   = lm->blocks[0]->n_heads;

        /* Create KV-caches for each layer.
         * IMPORTANT: allocate pointer array from params pool — scratch
         * gets reset between iterations and caches must survive. */
        kv_cache **caches = mem_params_alloc((size_t)lm->n_layers * sizeof(kv_cache*), NULL);
        int max_seq = max_len;
        for (int i = 0; i < lm->n_layers; i++) {
            caches[i] = kv_cache_create(B, H, max_seq, d_k);
        }

        /* Process prompt tokens one-by-one to populate cache */
        tensor *single_id_data = tensor_zeros_data(1, (int[]){1});  /* 1D int tensor */

        for (int p = 0; p < prompt_len; p++) {
            /* Set single token ID */
            ((int*)single_id_data->data)[0] = output[p];

            /* Embed: [1] → [1, d_model], then reshape to [1, 1, d_model] */
            tensor *flat_emb = tensor_embedding(lm->embedding_table, single_id_data);
            tensor *h = tensor_reshape(flat_emb, 3, (int[]){1, 1, d_model});

            /* Pass through all transformer blocks with cache */
            for (int i = 0; i < lm->n_layers; i++) {
                h = transformer_block_forward_cached(lm->blocks[i], h, caches[i]);
            }

            /* Final norm + lm_head (only needed for last prompt token's logits) */
            if (p == prompt_len - 1) {
                h = tensor_layer_norm(h, lm->norm_weight, lm->norm_bias, 1e-5f);
                tensor *logits = linear_forward(lm->lm_head, h);  /* [1, 1, vocab] */
                float *ld = tensor_data_ptr(logits);

                /* Copy last logits row before resetting scratch */
                float *last_logit_buf = mem_data_alloc((size_t)vocab_size * sizeof(float), NULL);
                memcpy(last_logit_buf, ld, (size_t)vocab_size * sizeof(float));
                mem_pool_reset(_mem_pool_scratch());

                int next_id;
                if (temperature == 0.0f) {
                    next_id = _argmax(last_logit_buf, vocab_size);
                } else {
                    next_id = _sample_with_temp(last_logit_buf, vocab_size, temperature);
                }

                output[cur_len++] = next_id;

                if (next_id == 258) goto done_generate;  /* EOS */
            } else {
                /* Free scratch between non-last prompt tokens */
                mem_pool_reset(_mem_pool_scratch());
            }
        }

        /* Generation loop: one token at a time using cache */
        while (cur_len < max_len) {
            ((int*)single_id_data->data)[0] = output[cur_len - 1];

            tensor *flat_emb = tensor_embedding(lm->embedding_table, single_id_data);
            tensor *h = tensor_reshape(flat_emb, 3, (int[]){1, 1, d_model});

            for (int i = 0; i < lm->n_layers; i++) {
                h = transformer_block_forward_cached(lm->blocks[i], h, caches[i]);
            }

            h = tensor_layer_norm(h, lm->norm_weight, lm->norm_bias, 1e-5f);
            tensor *logits = linear_forward(lm->lm_head, h);
            float *ld = tensor_data_ptr(logits);

            /* Copy last logits row before resetting scratch */
            float *last_logit_buf = mem_data_alloc((size_t)vocab_size * sizeof(float), NULL);
            memcpy(last_logit_buf, ld, (size_t)vocab_size * sizeof(float));
            mem_pool_reset(_mem_pool_scratch());

            int next_id;
            if (temperature == 0.0f) {
                next_id = _argmax(last_logit_buf, vocab_size);
            } else {
                next_id = _sample_with_temp(last_logit_buf, vocab_size, temperature);
            }

            output[cur_len++] = next_id;
            if (next_id == 258) goto done_generate;
        }

done_generate:
        ;

    } else {
        /* ── No cache path: full forward each step ── */
        while (cur_len < max_len) {
            /* Build tensor from current output buffer */
            tensor *ids_tensor = tensor_zeros_data(2, (int[]){1, cur_len});
            memcpy(ids_tensor->data, output, (size_t)cur_len * sizeof(int));

            /* Full forward pass */
            tensor *logits = decoder_lm_forward(lm, ids_tensor);  /* [1, cur_len, vocab] */

            /* Copy last token's logits before resetting scratch */
            float *ld = tensor_data_ptr(logits);
            float *last_logits = ld + (cur_len - 1) * vocab_size;
            float *last_logit_buf = mem_data_alloc((size_t)vocab_size * sizeof(float), NULL);
            memcpy(last_logit_buf, last_logits, (size_t)vocab_size * sizeof(float));

            /* Reset scratch — each forward pass allocates huge activations */
            mem_pool_reset(_mem_pool_scratch());

            int next_id;
            if (temperature == 0.0f) {
                next_id = _argmax(last_logit_buf, vocab_size);
            } else {
                next_id = _sample_with_temp(last_logit_buf, vocab_size, temperature);
            }

            output[cur_len++] = next_id;
            if (next_id == 258) goto done_nocache;
        }
done_nocache:
        ;
    }

    /* Restore grad mode */
    dnn_no_grad_exit(no_grad_ctx);

    *n_out = cur_len;
    return output;
}


/* ── Parameter count ── */

long long transformer_block_num_parameters(transformer_block *block) {
    assert(block);
    long long n = 0;
    n += linear_num_parameters(block->q_proj);
    n += linear_num_parameters(block->k_proj);
    n += linear_num_parameters(block->v_proj);
    n += linear_num_parameters(block->out_proj);
    n += tensor_numel(block->attn_norm_weight);
    n += tensor_numel(block->attn_norm_bias);
    n += tensor_numel(block->ffn_norm_weight);
    n += tensor_numel(block->ffn_norm_bias);
    n += swiglu_ffn_num_parameters(block->ffn);
    return n;
}

long long decoder_lm_num_parameters(decoder_lm *lm) {
    assert(lm);
    long long n = 0;
    n += tensor_numel(lm->embedding_table);
    for (int i = 0; i < lm->n_layers; i++)
        n += transformer_block_num_parameters(lm->blocks[i]);
    n += tensor_numel(lm->norm_weight);
    n += tensor_numel(lm->norm_bias);
    n += linear_num_parameters(lm->lm_head);
    return n;
}


/* ── RoPE position encoding ── */

void decoder_lm_enable_rope(decoder_lm *lm, int max_seq_len, float base) {
    assert(lm && max_seq_len > 0);

    int d_k = lm->blocks[0]->d_k;
    assert(d_k % 2 == 0 && "RoPE requires even head dimension");

    /* Init frequency tables in params pool */
    tensor *freqs_cos, *freqs_sin;
    tensor_rope_freqs_init(&freqs_cos, &freqs_sin, d_k, max_seq_len, base);

    /* Assign to every block */
    for (int i = 0; i < lm->n_layers; i++) {
        lm->blocks[i]->freqs_cos = freqs_cos;
        lm->blocks[i]->freqs_sin = freqs_sin;
    }
}
