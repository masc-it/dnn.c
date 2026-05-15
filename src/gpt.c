#include "gpt.h"
#include "rope.h"      /* tensor_rope_freqs_init */
#include "ops.h"       /* tensor_cross_entropy */
#include "pool.h"      /* mem_pool_reset / mark / release */
#include "autograd.h"  /* dnn_backward, dnn_no_grad_enter/exit */
#include "tokenizer.h" /* TOKENIZER_EOS_ID */
#include <assert.h>
#include <string.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>

/* Forward declaration */
static tensor *_lm_head_forward(struct mem_pool *scratch, decoder_lm *lm,
                                 const tensor *h);

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

    /* LM head: d_model → vocab_size (weight tied to embedding — share data via
     * embed->weight used with BLAS CblasTrans path instead of non-contig view) */
    lm->lm_head = _mem_pool_alloc(params_pool, sizeof(linear), NULL);
    module_init(&lm->lm_head->base, params_pool, "linear");
    lm->lm_head->in_features  = d_model;
    lm->lm_head->out_features = vocab_size;
    lm->lm_head->weight       = NULL;  /* not used — forward uses embed->weight */
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
    tensor *logits = _lm_head_forward(scratch, lm, h);

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
        memcpy(td + b * (N - 1), id + b * N + 1, (size_t)(N - 1) * sizeof(int));
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

/* ── LM head forward (tied embedding, BLAS transposed path) ── */

static tensor *_lm_head_forward(struct mem_pool *scratch, decoder_lm *lm,
                                 const tensor *h) {
    /* h @ embed_weight^T + bias.
     * embed_weight is [vocab_size, d_model] contiguous — BLAS CblasTrans.
     * Avoids non-contiguous transposed view that would skip BLAS entirely. */
    return tensor_matmul_add(scratch, h, lm->embed->weight, 1, lm->lm_head->bias);
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

        /* IMPORTANT: allocate KV caches from data_pool, not scratch_pool.
         * The generation loop calls mem_pool_reset(scratch_pool) per token
         * to reclaim activations.  If caches lived in scratch_pool they'd
         * be wiped on every reset.  Allocate from data_pool and mark so
         * we can release them after generation completes.
         * Mark is taken AFTER output allocation, so releasing mark does
         * not free the generated token buffer. */
        size_t cache_mark = mem_pool_mark(data_pool);

        kv_cache **caches = _mem_pool_alloc(data_pool, (size_t)lm->n_layers * sizeof(kv_cache*), NULL);
        int max_seq = max_len;
        for (int i = 0; i < lm->n_layers; i++) {
            caches[i] = kv_cache_create(data_pool, B, H, max_seq, d_k);
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
                tensor *logits = _lm_head_forward(scratch_pool, lm, h);  /* [1, 1, vocab] */
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
            tensor *logits = _lm_head_forward(scratch_pool, lm, h);
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
        /* Release KV-caches (everything after cache_mark on data_pool).
         * output is before cache_mark so it survives. */
        mem_pool_release(data_pool, cache_mark);
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
    /* Weight may be NULL (tied lm_head) — skip. Bias always init. */
    if (l->weight && tensor_is_contiguous(l->weight)) {
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
