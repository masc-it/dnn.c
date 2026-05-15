#include "transformer.h"
#include "nn.h"
#include "norm.h"
#include "ops.h"
#include "attention.h"
#include "multihead.h"
#include "rope.h"
#include "pool.h"
#include "autograd.h"
#include "optim.h"
#include "tokenizer.h"
#include "simd.h"
#include <assert.h>
#include <string.h>
#include <math.h>
#include <stdio.h>
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

kv_cache *kv_cache_create(struct mem_pool *params_pool, int B, int H, int max_seq, int d_k) {
    kv_cache *kvc = _mem_pool_alloc(params_pool, sizeof(kv_cache), NULL);
    int shape[4] = {B, H, max_seq, d_k};
    kvc->k_cache = tensor_zeros(params_pool, 4, shape, 0);
    kvc->v_cache = tensor_zeros(params_pool, 4, shape, 0);
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

tensor *kv_cache_get_K(struct mem_pool *scratch, kv_cache *kvc) {
    assert(kvc && "kv_cache_get_K: NULL cache");
    assert(kvc->seq_len > 0 && "kv_cache_get_K: empty cache");
    return tensor_slice(scratch, kvc->k_cache, 2, 0, kvc->seq_len);
}

tensor *kv_cache_get_V(struct mem_pool *scratch, kv_cache *kvc) {
    assert(kvc && "kv_cache_get_V: NULL cache");
    assert(kvc->seq_len > 0 && "kv_cache_get_V: empty cache");
    return tensor_slice(scratch, kvc->v_cache, 2, 0, kvc->seq_len);
}

/* ── Transformer Block ── */

transformer_block *transformer_block_create(struct mem_pool *params_pool, int d_model, int n_heads, int d_k,
                                             int intermediate_size) {
    assert(d_model > 0 && n_heads > 0 && d_k > 0 && intermediate_size > 0);
    assert(d_model == n_heads * d_k && "d_model must equal n_heads * d_k");

    transformer_block *block = _mem_pool_alloc(params_pool, sizeof(transformer_block), NULL);
    module_init(&block->base, params_pool, "transformer_block");

    block->d_model  = d_model;
    block->n_heads  = n_heads;
    block->d_k      = d_k;

    block->q_proj   = linear_create(params_pool, d_model, n_heads * d_k);
    module_add_child(&block->base, "q_proj", &block->q_proj->base);
    block->k_proj   = linear_create(params_pool, d_model, n_heads * d_k);
    module_add_child(&block->base, "k_proj", &block->k_proj->base);
    block->v_proj   = linear_create(params_pool, d_model, n_heads * d_k);
    module_add_child(&block->base, "v_proj", &block->v_proj->base);
    block->out_proj = linear_create(params_pool, n_heads * d_k, d_model);
    module_add_child(&block->base, "out_proj", &block->out_proj->base);

    /* Pre-attention and pre-FFN layer norms */
    block->attn_norm = layer_norm_create(params_pool, d_model, 1e-5f);
    module_add_child(&block->base, "attn_norm", &block->attn_norm->base);
    block->ffn_norm = layer_norm_create(params_pool, d_model, 1e-5f);
    module_add_child(&block->base, "ffn_norm", &block->ffn_norm->base);

    block->ffn = swiglu_ffn_create(params_pool, d_model, intermediate_size);
    module_add_child(&block->base, "ffn", &block->ffn->base);

    /* RoPE disabled by default */
    block->freqs_cos = NULL;
    block->freqs_sin = NULL;

    return block;
}

tensor *transformer_block_forward(struct mem_pool *scratch, transformer_block *block, const tensor *x) {
    assert(block && x);
    assert(x->ndim >= 2);
    assert(x->shape[x->ndim - 1] == block->d_model);

    /* ── Attention sublayer (pre-norm) ── */
    tensor *residual = (tensor*)x;
    tensor *h = layer_norm_forward(scratch, block->attn_norm, x);

    /* QKV projections: [B, N, d_model] → [B, N, n_heads * d_k] */
    tensor *Q = linear_forward(scratch, block->q_proj, h);
    tensor *K = linear_forward(scratch, block->k_proj, h);
    tensor *V = linear_forward(scratch, block->v_proj, h);

    /* Split heads: [B, N, H*d_k] → [B, H, N, d_k] */
    tensor *Qh = tensor_split_heads(scratch, Q, block->n_heads);
    tensor *Kh = tensor_split_heads(scratch, K, block->n_heads);
    tensor *Vh = tensor_split_heads(scratch, V, block->n_heads);

    /* ── RoPE: apply rotary position encoding to Q, K ── */
    if (block->freqs_cos && block->freqs_sin) {
        /* Slice freq tables to match current sequence length */
        int N = Qh->shape[2];
        tensor *fc = tensor_slice(scratch, block->freqs_cos, 0, 0, N);
        tensor *fs = tensor_slice(scratch, block->freqs_sin, 0, 0, N);
        Qh = tensor_rope(scratch, Qh, fc, fs);
        Kh = tensor_rope(scratch, Kh, fc, fs);
    }

    /* Fused causal attention (no extra mask needed) */
    tensor *attn_out = tensor_attention(scratch, Qh, Kh, Vh, NULL);

    /* Merge heads: [B, H, N, d_k] → [B, N, H*d_k] */
    tensor *attn_merged = tensor_merge_heads(scratch, attn_out);

    /* Output projection: [B, N, H*d_k] → [B, N, d_model] */
    tensor *attn_proj = linear_forward(scratch, block->out_proj, attn_merged);

    /* First residual: x = x + attn_proj */
    tensor *x_after_attn = tensor_add(scratch, residual, attn_proj);

    /* ── FFN sublayer (pre-norm) ── */
    residual = x_after_attn;
    h = layer_norm_forward(scratch, block->ffn_norm, x_after_attn);

    tensor *ffn_out = swiglu_ffn_forward(scratch, block->ffn, h);

    /* Second residual: return x + ffn_out */
    return tensor_add(scratch, residual, ffn_out);
}

/* ── Decoder-only Language Model ── */

decoder_lm *decoder_lm_create(struct mem_pool *params_pool, int vocab_size, int d_model,
                               int n_layers, int n_heads, int d_k,
                               int intermediate_size) {
    assert(vocab_size > 0 && d_model > 0 && n_layers > 0);
    assert(n_heads > 0 && d_k > 0 && intermediate_size > 0);
    assert(d_model == n_heads * d_k && "d_model must equal n_heads * d_k");

    decoder_lm *lm = _mem_pool_alloc(params_pool, sizeof(decoder_lm), NULL);
    module_init(&lm->base, params_pool, "decoder_lm");

    lm->d_model    = d_model;
    lm->vocab_size = vocab_size;
    lm->n_layers   = n_layers;

    /* Embedding table */
    lm->embed = embedding_create(params_pool, vocab_size, d_model);
    module_add_child(&lm->base, "embed", &lm->embed->base);

    /* Transformer blocks */
    lm->blocks = _mem_pool_alloc(params_pool, n_layers * sizeof(transformer_block*), NULL);
    for (int i = 0; i < n_layers; i++) {
        lm->blocks[i] = transformer_block_create(params_pool, d_model, n_heads, d_k,
                                                   intermediate_size);
        /* Build names like "blocks.0", "blocks.1", ... */
        char name_buf[32];
        int r = snprintf(name_buf, sizeof(name_buf), "blocks.%d", i);
        (void)r;
        /* Names are stable pointers — pool-allocated copy */
        size_t len = (size_t)r + 1;
        char *name = _mem_pool_alloc(params_pool, len, name_buf);
        module_add_child(&lm->base, name, &lm->blocks[i]->base);
    }

    /* Final layer norm */
    lm->norm = layer_norm_create(params_pool, d_model, 1e-5f);
    module_add_child(&lm->base, "norm", &lm->norm->base);

    /* LM head: d_model → vocab_size (weight tied to embedding) */
    lm->lm_head = _mem_pool_alloc(params_pool, sizeof(linear), NULL);
    module_init(&lm->lm_head->base, params_pool, "linear");
    lm->lm_head->in_features  = d_model;
    lm->lm_head->out_features = vocab_size;
    /* weight = transpose(embedding) — persistent copy in params pool */
    {
        tensor *tmp = tensor_transpose(params_pool, lm->embed->weight, 0, 1);
        lm->lm_head->weight = _mem_pool_alloc(params_pool, sizeof(tensor), tmp);
        lm->lm_head->weight->pool = params_pool;
    }
    lm->lm_head->bias = tensor_zeros(params_pool, 1, (int[]){vocab_size}, 1);
    /* bias is the only param of lm_head — weight is tied to embedding
       (shares data buffer), so it's NOT registered as a separate param.
       The optimizer sees it via embedding->weight. */
    module_param(&lm->lm_head->base, "bias", lm->lm_head->bias);
    module_add_child(&lm->base, "lm_head", &lm->lm_head->base);

    return lm;
}

tensor *decoder_lm_forward(struct mem_pool *scratch, decoder_lm *lm, const tensor *input_ids) {
    assert(lm && input_ids);
    assert(input_ids->ndim == 2 && "decoder_lm_forward: input_ids must be 2D [B, N]");
    assert(input_ids->contiguous && "decoder_lm_forward: input_ids must be contiguous");

    int B = input_ids->shape[0];
    int N = input_ids->shape[1];
    int D = lm->d_model;

    /* Flatten [B, N] → [B*N] for embedding lookup */
    tensor *flat_ids = tensor_flatten(scratch, (tensor*)input_ids);  /* view, no data copy */

    /* Embed: [B*N] → [B*N, d_model] */
    tensor *h = embedding_forward(scratch, NULL, lm->embed, flat_ids);

    /* Reshape to [B, N, d_model] */
    h = tensor_reshape(scratch, h, 3, (int[]){B, N, D});

    /* Pass through all transformer blocks */
    for (int i = 0; i < lm->n_layers; i++) {
        h = transformer_block_forward(scratch, lm->blocks[i], h);
    }

    /* Final layer norm */
    h = layer_norm_forward(scratch, lm->norm, h);

    /* LM head: [B, N, d_model] → [B, N, vocab_size] */
    tensor *logits = linear_forward(scratch, lm->lm_head, h);

    return logits;
}

/* ── Training step ── */

tensor *decoder_lm_train_step(struct mem_pool *scratch_pool, struct mem_pool *data_pool,
                               decoder_lm *lm, const tensor *input_ids,
                               adamw_opt *opt, float grad_clip,
                               float *grad_norm_out) {
    assert(lm && input_ids && opt);
    assert(input_ids->ndim == 2 && "train_step: input_ids must be 2D [B, N]");
    assert(input_ids->contiguous && "train_step: input_ids must be contiguous");
    assert(input_ids->shape[1] >= 2 && "train_step: need N >= 2 for at least 1 target");

    int B = input_ids->shape[0];
    int N = input_ids->shape[1];

    /* ── Forward ── */
    tensor *logits = decoder_lm_forward(scratch_pool, lm, input_ids);  /* [B, N, vocab] */

    /* ── Shift: logits[:, :-1, :] predict input_ids[:, 1:] ── */
    tensor *logits_shifted = tensor_slice(scratch_pool, logits, 1, 0, N - 1);  /* [B, N-1, vocab] */

    /* Build target = input_ids[:, 1:], same int-into-float trick */
    tensor *target = tensor_zeros_data(data_pool, 2, (int[]){B, N - 1});
    int *td = (int *)target->data;
    int *id = (int *)input_ids->data;
    for (int b = 0; b < B; b++) {
        for (int n = 1; n < N; n++) {
            td[b * (N - 1) + (n - 1)] = id[b * N + n];
        }
    }

    /* ── Loss: cross-entropy over vocab dim ── */
    tensor *loss = tensor_cross_entropy(scratch_pool, logits_shifted, target, 2);

    /* ── Backward ── */
    dnn_backward(scratch_pool, loss);

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

tensor *transformer_block_forward_cached(struct mem_pool *scratch,
                                          transformer_block *block,
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
    tensor *h = layer_norm_forward(scratch, block->attn_norm, x);

    /* QKV projections */
    tensor *Q = linear_forward(scratch, block->q_proj, h);  /* [B, N_new, H*d_k] */
    tensor *K = linear_forward(scratch, block->k_proj, h);
    tensor *V = linear_forward(scratch, block->v_proj, h);

    /* Split heads */
    tensor *Qh = tensor_split_heads(scratch, Q, H);  /* [B, H, N_new, d_k] */
    tensor *Kh = tensor_split_heads(scratch, K, H);
    tensor *Vh = tensor_split_heads(scratch, V, H);

    /* ── RoPE: apply to Q, K with position offset ── */
    if (block->freqs_cos && block->freqs_sin) {
        /* New tokens are at positions [cache->seq_len, cache->seq_len + N_new)
         * Slice the freq tables to start at the current cache position. */
        int pos_offset = cache->seq_len;
        tensor *fc_slice = tensor_slice(scratch, block->freqs_cos, 0, pos_offset, N_new);
        tensor *fs_slice = tensor_slice(scratch, block->freqs_sin, 0, pos_offset, N_new);
        Qh = tensor_rope(scratch, Qh, fc_slice, fs_slice);
        Kh = tensor_rope(scratch, Kh, fc_slice, fs_slice);
    }

    /* Append new K/V to cache */
    kv_cache_append(cache, Kh, Vh);

    /* Get full cached K/V */
    tensor *K_full = kv_cache_get_K(scratch, cache);  /* [B, H, S, d_k] */
    tensor *V_full = kv_cache_get_V(scratch, cache);  /* [B, H, S, d_k] */

    int S = K_full->shape[2];
    float scale = 1.0f / sqrtf((float)d_k);

    /* Output allocation */
    tensor *attn_out = tensor_scratch(scratch, 4, (int[]){B, H, N_new, d_k}, 0);
    float *od = (float*)attn_out->data;
    float *qd = (float*)Qh->data;
    float *kd = (float*)K_full->data;
    float *vd = (float*)V_full->data;

    int q_sN = Qh->strides[2];
    int k_sN = K_full->strides[2];
    int v_sN = V_full->strides[2];
    int o_sN = attn_out->strides[2];

    /* Temp scores [N_new, S] */
    float *scores = _mem_pool_alloc(scratch, (size_t)N_new * S * sizeof(float), NULL);

    for (int b = 0; b < B; b++) {
        for (int h = 0; h < H; h++) {
            float *q_slice = qd + Qh->offset
                           + b * Qh->strides[0]
                           + h * Qh->strides[1];
            float *k_slice = kd + K_full->offset
                           + b * K_full->strides[0]
                           + h * K_full->strides[1];
            float *v_slice = vd + V_full->offset
                           + b * V_full->strides[0]
                           + h * V_full->strides[1];
            float *o_slice = od + attn_out->offset
                           + b * attn_out->strides[0]
                           + h * attn_out->strides[1];

            /* scores = Q @ K^T * scale  [N_new, S] */
#if NO_CBLAS
            for (int i = 0; i < N_new; i++)
                for (int j = 0; j < S; j++) {
                    float sum = 0.0f;
                    for (int kk = 0; kk < d_k; kk++)
                        sum += q_slice[i * q_sN + kk] * k_slice[j * k_sN + kk];
                    scores[i * S + j] = sum * scale;
                }
#else
            cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans,
                        N_new, S, d_k, scale, q_slice, q_sN, k_slice, k_sN,
                        0.0f, scores, S);
#endif

            /* ── Causal mask: row i attends only to positions ≤ old_seq + i ── */
            {
                int old_seq = S - N_new;
                for (int i = 0; i < N_new; i++) {
                    float *row = scores + i * S;
                    for (int j = old_seq + i + 1; j < S; j++)
                        row[j] = -INFINITY;
                }
            }

            /* Softmax over last dim — causal mask applied above
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
                        sum += scores[i * S + kk] * v_slice[kk * v_sN + j];
                    o_slice[i * o_sN + j] = sum;
                }
#else
            cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
                        N_new, d_k, S, 1.0f, scores, S, v_slice, v_sN,
                        0.0f, o_slice, o_sN);
#endif
        }
    }

    /* No autograd — generation is eval-only */

    /* Merge heads: [B, H, N_new, d_k] → [B, N_new, H*d_k] */
    tensor *attn_merged = tensor_merge_heads(scratch, attn_out);

    /* Output projection */
    tensor *attn_proj = linear_forward(scratch, block->out_proj, attn_merged);

    /* First residual */
    tensor *x_after_attn = tensor_add(scratch, x, attn_proj);

    /* ── FFN sublayer (pre-norm) ── */
    h = layer_norm_forward(scratch, block->ffn_norm, x_after_attn);
    tensor *ffn_out = swiglu_ffn_forward(scratch, block->ffn, h);

    /* Second residual */
    return tensor_add(scratch, x_after_attn, ffn_out);
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

int *decoder_lm_generate(struct mem_pool *scratch_pool, struct mem_pool *data_pool,
                          decoder_lm *lm, const tensor *prompt_ids,
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
    int *output = _mem_pool_alloc(data_pool, (size_t)max_len * sizeof(int), NULL);
    memcpy(output, prompt_ids->data, (size_t)prompt_len * sizeof(int));
    int cur_len = prompt_len;

    /* Enter no-grad mode for generation */
    dnn_grad_ctx no_grad_ctx = dnn_no_grad_enter();

    if (use_cache && lm->n_layers > 0) {
        /* ── KV-cache path ── */
        int d_k = lm->blocks[0]->d_k;
        int H   = lm->blocks[0]->n_heads;

        size_t cache_mark = mem_pool_mark(scratch_pool);

        /* Create KV-caches for each layer.
         * IMPORTANT: allocate pointer array from params pool — scratch
         * gets reset between iterations and caches must survive. */
        kv_cache **caches = _mem_pool_alloc(scratch_pool, (size_t)lm->n_layers * sizeof(kv_cache*), NULL);
        int max_seq = max_len;
        for (int i = 0; i < lm->n_layers; i++) {
            caches[i] = kv_cache_create(scratch_pool, B, H, max_seq, d_k);
        }

        /* Process prompt tokens one-by-one to populate cache */
        tensor *single_id_data = tensor_zeros_data(data_pool, 1, (int[]){1});  /* 1D int tensor */

        for (int p = 0; p < prompt_len; p++) {
            /* Set single token ID */
            ((int*)single_id_data->data)[0] = output[p];

            /* Embed: [1] → [1, d_model], then reshape to [1, 1, d_model] */
            tensor *flat_emb = embedding_forward(scratch_pool, NULL, lm->embed, single_id_data);
            tensor *h = tensor_reshape(scratch_pool, flat_emb, 3, (int[]){1, 1, d_model});

            /* Pass through all transformer blocks with cache */
            for (int i = 0; i < lm->n_layers; i++) {
                h = transformer_block_forward_cached(scratch_pool, lm->blocks[i], h, caches[i]);
            }

            /* Final norm + lm_head (only needed for last prompt token's logits) */
            if (p == prompt_len - 1) {
                h = layer_norm_forward(scratch_pool, lm->norm, h);
                tensor *logits = linear_forward(scratch_pool, lm->lm_head, h);  /* [1, 1, vocab] */
                float *ld = tensor_data_ptr(logits);

                /* Copy last logits row before resetting scratch */
                float *last_logit_buf = _mem_pool_alloc(data_pool, (size_t)vocab_size * sizeof(float), NULL);
                memcpy(last_logit_buf, ld, (size_t)vocab_size * sizeof(float));
                mem_pool_reset(scratch_pool);

                int next_id;
                if (temperature == 0.0f) {
                    next_id = _argmax(last_logit_buf, vocab_size);
                } else {
                    next_id = _sample_with_temp(last_logit_buf, vocab_size, temperature);
                }

                output[cur_len++] = next_id;

                if (next_id == TOKENIZER_EOS_ID) goto done_generate;
            } else {
                /* Free scratch between non-last prompt tokens */
                mem_pool_reset(scratch_pool);
            }
        }

        /* Generation loop: one token at a time using cache */
        while (cur_len < max_len) {
            ((int*)single_id_data->data)[0] = output[cur_len - 1];

            tensor *flat_emb = embedding_forward(scratch_pool, NULL, lm->embed, single_id_data);
            tensor *h = tensor_reshape(scratch_pool, flat_emb, 3, (int[]){1, 1, d_model});

            for (int i = 0; i < lm->n_layers; i++) {
                h = transformer_block_forward_cached(scratch_pool, lm->blocks[i], h, caches[i]);
            }

            h = layer_norm_forward(scratch_pool, lm->norm, h);
            tensor *logits = linear_forward(scratch_pool, lm->lm_head, h);
            float *ld = tensor_data_ptr(logits);

            /* Copy last logits row before resetting scratch */
            float *last_logit_buf = _mem_pool_alloc(data_pool, (size_t)vocab_size * sizeof(float), NULL);
            memcpy(last_logit_buf, ld, (size_t)vocab_size * sizeof(float));
            mem_pool_reset(scratch_pool);

            int next_id;
            if (temperature == 0.0f) {
                next_id = _argmax(last_logit_buf, vocab_size);
            } else {
                next_id = _sample_with_temp(last_logit_buf, vocab_size, temperature);
            }

            output[cur_len++] = next_id;
            if (next_id == TOKENIZER_EOS_ID) goto done_generate;
        }

done_generate:
        mem_pool_release(scratch_pool, cache_mark);
        ;

    } else {
        /* ── No cache path: full forward each step ── */
        while (cur_len < max_len) {
            /* Build tensor from current output buffer */
            tensor *ids_tensor = tensor_zeros_data(data_pool, 2, (int[]){1, cur_len});
            memcpy(ids_tensor->data, output, (size_t)cur_len * sizeof(int));

            /* Full forward pass */
            tensor *logits = decoder_lm_forward(scratch_pool, lm, ids_tensor);  /* [1, cur_len, vocab] */

            /* Copy last token's logits before resetting scratch */
            float *ld = tensor_data_ptr(logits);
            float *last_logits = ld + (cur_len - 1) * vocab_size;
            float *last_logit_buf = _mem_pool_alloc(data_pool, (size_t)vocab_size * sizeof(float), NULL);
            memcpy(last_logit_buf, last_logits, (size_t)vocab_size * sizeof(float));

            /* Reset scratch — each forward pass allocates huge activations */
            mem_pool_reset(scratch_pool);

            int next_id;
            if (temperature == 0.0f) {
                next_id = _argmax(last_logit_buf, vocab_size);
            } else {
                next_id = _sample_with_temp(last_logit_buf, vocab_size, temperature);
            }

            output[cur_len++] = next_id;
            if (next_id == TOKENIZER_EOS_ID) goto done_nocache;
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
    return module_num_parameters(&block->base);
}

long long decoder_lm_num_parameters(decoder_lm *lm) {
    return module_num_parameters(&lm->base);
}


/* ── Weight initialization (GPT-2 style) ── */

/* Box-Muller normal random sample */
static float _randn(void) {
    float u1 = (float)rand() / (float)RAND_MAX;
    float u2 = (float)rand() / (float)RAND_MAX;
    return sqrtf(-2.0f * logf(u1 + 1e-10f)) * cosf(6.283185307179586f * u2);
}



static void _init_linear(linear *l, float std) {
    /* Weight may be a transposed view (tied weights) — skip if non-contiguous.
       Bias is always contiguous. */
    if (tensor_is_contiguous(l->weight)) {
        int nw = tensor_numel(l->weight);
        float *wd = tensor_data_ptr(l->weight);
        for (int i = 0; i < nw; i++) wd[i] = _randn() * std;
    }
    memset(tensor_data_ptr(l->bias), 0, (size_t)tensor_numel(l->bias) * sizeof(float));
}

/* Recursively walk module tree, init linear weights.
   Residual branches (out_proj, down_proj) get scaled std. */
static void _init_module_weights(module *m, float std, float residual_std) {
    for (module_item *item = m->items_head; item; item = item->next) {
        if (item->kind == MODULE_ITEM_CHILD) {
            module *child = item->as.child;
            if (strcmp(child->type_name, "linear") == 0) {
                float s = (strstr(item->name, "out_proj") ||
                           strstr(item->name, "down_proj"))
                          ? residual_std : std;
                _init_linear((linear *)child, s);
            } else {
                _init_module_weights(child, std, residual_std);
            }
        }
    }
}

void decoder_lm_init_weights(decoder_lm *lm) {
    assert(lm);
    float std = 0.02f;
    float residual_std = std / sqrtf(2.0f * (float)lm->n_layers);

    /* Embedding table: Normal(0, std) */
    int ne = tensor_numel(lm->embed->weight);
    float *ed = tensor_data_ptr(lm->embed->weight);
    for (int i = 0; i < ne; i++) ed[i] = _randn() * std;

    /* Walk all children recursively — inits all linears (weight + bias).
       lm_head weight is a tied view of embed->weight (already inited),
       so _init_linear skips it and only zeroes the bias. */
    _init_module_weights(&lm->base, std, residual_std);

    /* Layer norm γ=1, β=0 — already correct after decoder_lm_create, skip */
}


/* ── RoPE position encoding ── */

void decoder_lm_enable_rope(struct mem_pool *params_pool, decoder_lm *lm, int max_seq_len, float base) {
    assert(lm && max_seq_len > 0);

    int d_k = lm->blocks[0]->d_k;
    assert(d_k % 2 == 0 && "RoPE requires even head dimension");

    /* Init frequency tables in params pool */
    tensor *freqs_cos, *freqs_sin;
    tensor_rope_freqs_init(params_pool, &freqs_cos, &freqs_sin, d_k, max_seq_len, base);

    /* Assign to every block */
    for (int i = 0; i < lm->n_layers; i++) {
        lm->blocks[i]->freqs_cos = freqs_cos;
        lm->blocks[i]->freqs_sin = freqs_sin;
    }
}
