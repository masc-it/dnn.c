/* ══════════════════════════════════════════════════════════════════
 *  vlm_init_probe — analyze VLM init scheme analytically
 *
 *  Creates model, inits weights, runs forward+backward with
 *  synthetic data and measures activation/gradient statistics
 *  per component to detect scale mismatches.
 * ══════════════════════════════════════════════════════════════════ */

#include "dnn.h"
#include "gpt.h"
#include "vlm.h"
#include "optim.h"
#include "tokenizer.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <sys/param.h>

/* ── Model config (matches imagenet_vlm.c) ── */
#define IMG_H           224
#define IMG_W           224
#define IMG_C             3
#define PATCH_SIZE       16
#define N_IMG_TOK       196
#define D_MODEL         128
#define N_LAYERS          2
#define N_HEADS           2
#define D_K             64
#define INTERMEDIATE   256
#define VOCAB_SIZE      261
#define BATCH_SIZE       32
#define TMAX             48     /* labels ~5-30 bytes + BOS/EOS */

/* ── Stats helpers ── */

typedef struct { double mean; double std; double min; double max; double rms; } tensor_stat;

static tensor_stat compute_stat(const float *data, int n) {
    tensor_stat s = {0,0,0,0,0};
    if (n <= 0) return s;
    double sum = 0.0, sum2 = 0.0;
    s.min = INFINITY; s.max = -INFINITY;
    for (int i = 0; i < n; i++) {
        double v = data[i];
        sum += v;
        sum2 += v*v;
        if (v < s.min) s.min = v;
        if (v > s.max) s.max = v;
    }
    s.mean = sum / n;
    s.std = sqrt(sum2/n - s.mean*s.mean);
    s.rms = sqrt(sum2/n);
    return s;
}

static void print_shape(const tensor *t) {
    printf("[");
    for (int i = 0; i < t->ndim; i++) {
        if (i > 0) printf(",");
        printf("%d", t->shape[i]);
    }
    printf("]");
}

static void print_tstat(const char *label, tensor *t) {
    int n = tensor_numel(t);
    float *d = tensor_data_ptr(t);
    tensor_stat s = compute_stat(d, n);
    printf("  %-30s ", label);
    print_shape(t);
    printf(" mean=%+.4e std=%.4e rms=%.4e min=%.4e max=%.4e\n",
           s.mean, s.std, s.rms, s.min, s.max);
}

static void print_grad_stat(const char *label, tensor *t) {
    float *g = tensor_grad(t);
    if (!g) { printf("  %-30s (no grad)\n", label); return; }
    int n = tensor_numel(t);
    tensor_stat s = compute_stat(g, n);
    printf("  %-30s ", label);
    print_shape(t);
    printf(" mean=%+.4e std=%.4e rms=%.4e max|g|=%+.4e\n",
           s.mean, s.std, s.rms, s.max);
}

static void param_stat(const char *name, tensor *p) {
    int n = tensor_numel(p);
    float *d = tensor_data_ptr(p);
    tensor_stat s = compute_stat(d, n);
    printf("  PARAM %-25s ", name);
    print_shape(p);
    printf(" mean=%+.4e std=%.4e rms=%.4e\n",
           s.mean, s.std, s.rms);
}

/* ── Walk module tree and print all param stats ── */

static void walk_params(module *m, const char *prefix) {
    for (module_item *item = m->items_head; item; item = item->next) {
        char name[256];
        if (item->kind == MODULE_ITEM_PARAM) {
            snprintf(name, sizeof(name), "%s%s", prefix, item->name);
            param_stat(name, item->as.param);
        } else if (item->kind == MODULE_ITEM_CHILD) {
            snprintf(name, sizeof(name), "%s%s.", prefix, item->name);
            walk_params(item->as.child, name);
        }
    }
}

/* ── Forward+backward and collect activation/grad stats ── */

static void run_probe(vision_lm *vlm, struct mem_pool *scratch,
                       struct mem_pool *data) {
    int B = BATCH_SIZE;
    int T = TMAX;
    int I = vlm->n_img_tokens;

    /* ── Create synthetic data ── */
    tensor *images = tensor_zeros_data(data, 4, (int[]){B, IMG_C, IMG_H, IMG_W});
    {
        int npx = tensor_numel(images);
        float *d = tensor_data_ptr(images);
        for (int i = 0; i < npx; i++)
            d[i] = ((float)rand() / RAND_MAX) * 2.0f - 1.0f;
    }

    tensor *input_ids = tensor_zeros_data(data, 2, (int[]){B, T});
    tensor *target_ids = tensor_zeros_data(data, 2, (int[]){B, T});
    tensor *loss_mask = tensor_zeros_data(data, 2, (int[]){B, T});
    int text_lens[B];
    for (int b = 0; b < B; b++) {
        int len = (rand() % 20) + 5;
        if (len > T - 2) len = T - 2;
        int *inp = (int *)tensor_data_ptr(input_ids) + (long)b * T;
        int *tgt = (int *)tensor_data_ptr(target_ids) + (long)b * T;
        float *msk = tensor_data_ptr(loss_mask) + (long)b * T;
        inp[0] = TOKENIZER_BOS_ID;
        for (int t = 0; t < len; t++) {
            int tok = (rand() % 95) + 32;
            inp[t + 1] = tok;
            tgt[t]     = tok;
            msk[t]     = 1.0f;
        }
        tgt[len] = TOKENIZER_EOS_ID;
        msk[len] = 1.0f;
        text_lens[b] = len + 1;
    }

    /* ══════════════════════════════════════════════════════════
     *  1. FORWARD: activation statistics at each stage
     * ══════════════════════════════════════════════════════════ */
    printf("\n═══ 1. Forward activation stats ═══\n");

    tensor *img_feat = vision_lm_image_embeds(scratch, vlm, images);
    printf("\n--- After patch_embed + transpose [B, I, D] ---\n");
    print_tstat("img_feat (raw conv)", img_feat);

    tensor *img_normed = rms_norm_forward(scratch, vlm->image_norm, img_feat);
    printf("\n--- After image_norm (RMSNorm) ---\n");
    print_tstat("img_feat (normed)", img_normed);

    tensor *img_with_pos = NULL;
    if (vlm->use_image_pos && vlm->image_pos) {
        img_with_pos = tensor_add(scratch, img_normed, vlm->image_pos);
        printf("\n--- After adding image_pos ---\n");
        print_tstat("img_feat (pos)", img_with_pos);
    } else {
        img_with_pos = img_normed;
    }

    tensor *txt_feat = decoder_lm_token_embeds(scratch, vlm->lm, input_ids);
    printf("\n--- After text embed ---\n");
    print_tstat("txt_feat", txt_feat);

    tensor *embeds = tensor_cat(scratch, img_with_pos, txt_feat, 1);
    printf("\n--- After concat [B, I+T, D] ---\n");
    printf("  Full seq shape: [%d, %d, %d]\n", B, I+T, D_MODEL);
    print_tstat("embeds (full seq)", embeds);

    /* Extract image portion and text portion separately.
     * Must contiguous() before accessing data ptr in print_tstat. */
    tensor *emb_img_part = tensor_contiguous(scratch, tensor_slice(scratch, embeds, 1, 0, I));
    tensor *emb_txt_part = tensor_contiguous(scratch, tensor_slice(scratch, embeds, 1, I, T));
    printf("  Image part only:\n");
    print_tstat("  embeds[img]", emb_img_part);
    printf("  Text part only:\n");
    print_tstat("  embeds[txt]", emb_txt_part);

    /* Forward through transformer layers */
    printf("\n--- Per-layer hidden RMS ---\n");
    tensor *h = embeds;
    for (int i = 0; i < vlm->lm->n_layers; i++) {
        h = transformer_block_forward_ex(scratch, vlm->lm->blocks[i], h,
                                         ATTENTION_PREFIX_LM, I, NULL);
        int nh = tensor_numel(h);
        float *hd = tensor_data_ptr(h);
        double rms = 0.0;
        for (int j = 0; j < nh; j++) rms += (double)hd[j] * (double)hd[j];
        rms = sqrt(rms / nh);
        printf("  layer %d after block: rms=%.4f\n", i, rms);
    }
    h = rms_norm_forward(scratch, vlm->lm->norm, h);
    {
        int nh = tensor_numel(h);
        float *hd = tensor_data_ptr(h);
        double rms = 0.0;
        for (int j = 0; j < nh; j++) rms += (double)hd[j] * (double)hd[j];
        rms = sqrt(rms / nh);
        printf("  after final norm: rms=%.4f\n", rms);
    }

    /* Now compute loss to get a signal for backward */
    printf("\n--- Loss with synthetic data ---\n");
    tensor *logits_full = decoder_lm_forward_embeds_ex(scratch, vlm->lm, embeds,
                                                       ATTENTION_PREFIX_LM, I, NULL);
    tensor *logits_text = tensor_contiguous(scratch, tensor_slice(scratch, logits_full, 1, I, T));
    tensor *loss = tensor_cross_entropy_masked(scratch, logits_text, target_ids, loss_mask, 2);
    float loss_val = tensor_data_ptr(loss)[0];
    double random_chance = log((double)VOCAB_SIZE);
    printf("  Loss (at init): %.6f  (random chance ~ %.2f)\n",
           loss_val, random_chance);

    /* ══════════════════════════════════════════════════════════
     *  2. BACKWARD: gradient statistics
     * ══════════════════════════════════════════════════════════ */
    printf("\n═══ 2. Backward gradient stats ═══\n");

    /* We need fresh forward for backward with grad tracking.
     * Reset pools and redo with requires_grad. */
    mem_pool_reset(scratch);
    mem_pool_reset(data);

    /* Recreate tensors */
    images = tensor_zeros_data(data, 4, (int[]){B, IMG_C, IMG_H, IMG_W});
    {
        int npx = tensor_numel(images);
        float *d = tensor_data_ptr(images);
        for (int i = 0; i < npx; i++)
            d[i] = ((float)rand() / RAND_MAX) * 2.0f - 1.0f;
    }
    input_ids = tensor_zeros_data(data, 2, (int[]){B, T});
    target_ids = tensor_zeros_data(data, 2, (int[]){B, T});
    loss_mask = tensor_zeros_data(data, 2, (int[]){B, T});
    for (int b = 0; b < B; b++) {
        int len = (rand() % 20) + 5;
        if (len > T - 2) len = T - 2;
        int *inp = (int *)tensor_data_ptr(input_ids) + (long)b * T;
        int *tgt = (int *)tensor_data_ptr(target_ids) + (long)b * T;
        float *msk = tensor_data_ptr(loss_mask) + (long)b * T;
        inp[0] = TOKENIZER_BOS_ID;
        for (int t = 0; t < len; t++) {
            int tok = (rand() % 95) + 32;
            inp[t + 1] = tok;
            tgt[t]     = tok;
            msk[t]     = 1.0f;
        }
        tgt[len] = TOKENIZER_EOS_ID;
        msk[len] = 1.0f;
        text_lens[b] = len + 1;
    }

    /* Get all params, zero their grads */
    int n_params;
    tensor **all_params = module_parameters(&vlm->base, &n_params);
    for (int i = 0; i < n_params; i++) {
        if (tensor_grad(all_params[i])) {
            int ng = tensor_numel(all_params[i]);
            memset(tensor_grad(all_params[i]), 0, (size_t)ng * sizeof(float));
        }
    }

    /* Forward+backward */
    tensor *loss2 = vision_lm_loss_padded(scratch, vlm, images, input_ids,
                                          target_ids, loss_mask, text_lens);
    dnn_backward(scratch, loss2);
    printf("  Loss (bwd pass): %.6f\n", tensor_data_ptr(loss2)[0]);

    /* Identify key params */
    tensor *pw = module_find_param(&vlm->base, "patch_embed.weight");
    tensor *pb = module_find_param(&vlm->base, "patch_embed.bias");
    tensor *pn = module_find_param(&vlm->base, "image_norm.weight");
    tensor *pp = module_find_param(&vlm->base, "image_pos");
    tensor *ew = module_find_param(&vlm->base, "lm.embed.weight");
    tensor *lb = module_find_param(&vlm->base, "lm.lm_head.bias");

    /* Print gradient stat per image param */
    printf("\n--- Image pathway grads ---\n");
    if (pw && tensor_grad(pw)) print_grad_stat("patch_embed.weight", pw);
    if (pb && tensor_grad(pb)) print_grad_stat("patch_embed.bias", pb);
    if (pn && tensor_grad(pn)) print_grad_stat("image_norm.weight", pn);
    if (pp && tensor_grad(pp)) print_grad_stat("image_pos", pp);

    /* LM embed */
    printf("\n--- LM embedding grads ---\n");
    if (ew && tensor_grad(ew)) print_grad_stat("lm.embed.weight", ew);
    if (lb && tensor_grad(lb)) print_grad_stat("lm.lm_head.bias", lb);

    /* Group grad norm breakdown */
    printf("\n--- Gradient norm breakdown ---\n");
    double total_grad_sq = 0.0;
    double img_grad_sq = 0.0;
    double embed_grad_sq = 0.0;
    double lm_grad_sq = 0.0;

    for (int i = 0; i < n_params; i++) {
        tensor *p = all_params[i];
        float *g = tensor_grad(p);
        if (!g) continue;
        int ng = tensor_numel(p);
        double sq = 0.0;
        for (int j = 0; j < ng; j++) sq += (double)g[j] * (double)g[j];
        total_grad_sq += sq;
        if (p == pw || p == pb || p == pn || p == pp) img_grad_sq += sq;
        else if (p == ew) embed_grad_sq += sq;
        else lm_grad_sq += sq;
    }

    double total_gn = sqrt(total_grad_sq);
    double img_gn = sqrt(img_grad_sq);
    double embed_gn = sqrt(embed_grad_sq);
    double lm_gn = sqrt(lm_grad_sq);
    printf("  Total grad norm:         %.4e\n", total_gn);
    if (total_gn > 1e-30) {
        printf("  Image pathway:           %.4e (%.1f%%)\n",
               img_gn, 100.0 * img_grad_sq / total_grad_sq);
        printf("  Embedding table:         %.4e (%.1f%%)\n",
               embed_gn, 100.0 * embed_grad_sq / total_grad_sq);
        printf("  LM other:                %.4e (%.1f%%)\n",
               lm_gn, 100.0 * lm_grad_sq / total_grad_sq);
    }

    /* Compute per-weight grad std ratios for LR multiplier recommendation */
    printf("\n--- Grad std per param (for LR mult analysis) ---\n");
    if (pw && tensor_grad(pw) && ew && tensor_grad(ew)) {
        int nw = tensor_numel(pw);
        int ne = tensor_numel(ew);
        tensor_stat gs_pw = compute_stat(tensor_grad(pw), nw);
        tensor_stat gs_ew = compute_stat(tensor_grad(ew), ne);
        printf("  patch_embed.weight grad std: %.4e\n", gs_pw.std);
        printf("  lm.embed.weight grad std:    %.4e\n", gs_ew.std);
        double ratio = (gs_ew.std > 1e-30) ? gs_pw.std / gs_ew.std : 0;
        printf("  patch/embed grad std ratio:  %.3f\n", ratio);
        printf("  With PATCH_LR_MULT=3.0:      effective ratio = %.3f\n", ratio * 3.0);
        if (ratio * 3.0 > 5.0)
            printf("  >>> WARNING: 3x may be too high\n");
        else if (ratio * 3.0 < 0.3)
            printf("  >>> WARNING: 3x may be too low\n");
        else
            printf("  >>> OK: 3x in reasonable range\n");
    }

    /* ══════════════════════════════════════════════════════════
     *  3. PARAMETER INIT STATS
     * ══════════════════════════════════════════════════════════ */
    printf("\n═══ 3. Init parameter statistics ═══\n");
    walk_params(&vlm->base, "");

    /* ══════════════════════════════════════════════════════════
     *  4. PARAMETER COUNT BREAKDOWN
     * ══════════════════════════════════════════════════════════ */
    printf("\n═══ 4. Parameter count breakdown ═══\n");
    long long tot_params = module_num_parameters(&vlm->base);
    long long img_count = 0;
    img_count += pw ? tensor_numel(pw) : 0;
    img_count += pb ? tensor_numel(pb) : 0;
    img_count += pn ? tensor_numel(pn) : 0;
    img_count += pp ? tensor_numel(pp) : 0;
    long long embed_count = ew ? tensor_numel(ew) : 0;

    printf("  Image pathway:        %lld (%.1f%%)\n",
           img_count, 100.0 * img_count / tot_params);
    if (pw) printf("    patch_embed.weight: %d\n", tensor_numel(pw));
    if (pb) printf("    patch_embed.bias:   %d\n", tensor_numel(pb));
    if (pn) printf("    image_norm.weight:  %d\n", tensor_numel(pn));
    if (pp) printf("    image_pos:          %d\n", tensor_numel(pp));
    printf("  lm.embed.weight:      %lld (%.1f%%)\n",
           embed_count, 100.0 * embed_count / tot_params);
    printf("  LM other:             %lld (%.1f%%)\n",
           tot_params - img_count - embed_count,
           100.0 * (tot_params - img_count - embed_count) / tot_params);
    printf("  TOTAL:                %lld (%.2fM)\n", tot_params, tot_params / 1e6);

    /* ══════════════════════════════════════════════════════════
     *  5. ANALYTICAL SCALE CHECKS
     * ══════════════════════════════════════════════════════════ */
    printf("\n═══ 5. Analytical scale checks ═══\n");

    /* 5a. Conv2d Xavier formula check */
    {
        int C = IMG_C, P = PATCH_SIZE, D = D_MODEL;
        double fan_in = (double)C * P * P;
        double fan_out = (double)D * P * P;
        printf("  [patch_embed.weight init formula]\n");
        printf("    fan_in (C*P*P)            = %.0f\n", fan_in);
        printf("    fan_out (D*P*P)           = %.0f\n", fan_out);
        printf("    Code formula: sqrt(2/(fan_in + D)) = sqrt(2/(%.0f+%d)) = %.4f\n",
               fan_in, D, sqrtf(2.0f / (float)(C*P*P + D)));
        printf("    Xavier normal: sqrt(2/(fan_in + fan_out)) = sqrt(2/(%.0f+%.0f)) = %.4f\n",
               fan_in, fan_out, sqrt(2.0 / (fan_in + fan_out)));
        printf("    Kaiming fan-in (no act):  sqrt(1/fan_in) = %.4f\n",
               sqrt(1.0 / fan_in));
        printf("    Kaiming fan-in (ReLU):    sqrt(2/fan_in) = %.4f\n",
               sqrt(2.0 / fan_in));
        printf("    Kaiming fan-in (GELU):    ~sqrt(1.2/fan_in) = %.4f\n",
               sqrt(1.2 / fan_in));

        /* Output variance */
        tensor_stat ws = compute_stat(tensor_data_ptr(pw), tensor_numel(pw));
        double actual_out_var = fan_in * ws.std * ws.std;
        printf("    Actual init std            = %.4f\n", ws.std);
        printf("    Expected conv output var   = fan_in * std^2 = %.4f\n", actual_out_var);
        printf("    >>> Ideal var ≈ 1 for signal preservation\n");
        if (fabs(actual_out_var - 1.0) > 0.5)
            printf("    >>> WARNING: output var=%.2f, far from 1\n", actual_out_var);
        else
            printf("    >>> OK: output var=%.2f near 1\n", actual_out_var);
    }

    /* 5b. Embedding vs patch scale at concat */
    {
        /* Re-run forward to get embeds for this section.
         * Ensure contiguous before tensor_data_ptr. */
        tensor *feat2 = vision_lm_image_embeds(scratch, vlm, images);
        tensor *nrm2 = tensor_contiguous(scratch, rms_norm_forward(scratch, vlm->image_norm, feat2));
        tensor *txt2 = tensor_contiguous(scratch, decoder_lm_token_embeds(scratch, vlm->lm, input_ids));

        float *id = tensor_data_ptr(nrm2);
        float *td = tensor_data_ptr(txt2);
        int ni = tensor_numel(nrm2);
        int nt = tensor_numel(txt2);
        tensor_stat is = compute_stat(id, ni);
        tensor_stat ts = compute_stat(td, nt);
        printf("\n  [Embedding vs Patch scale at concat]\n");
        printf("    Image part RMS:   %.4f\n", is.rms);
        printf("    Text part RMS:    %.4f\n", ts.rms);
        double ratio = (ts.rms > 1e-10) ? (is.rms / ts.rms) : 999.0;
        printf("    Image RMS / Text RMS: %.2f\n", ratio);
        if (ratio > 5.0 || ratio < 0.2)
            printf("    >>> WARNING: severe scale mismatch\n");
        else if (ratio > 2.0 || ratio < 0.5)
            printf("    >>> CAUTION: moderate scale mismatch (ratio=%.2f)\n", ratio);
        else
            printf("    >>> OK: scales within 2x\n");

        /* Theoretical check */
        double embed_var = (double)D_MODEL * 0.02 * 0.02;
        printf("    Expected text RMS ≈ sqrt(D*std^2) = sqrt(%d*%.4f^2) = %.4f\n",
               D_MODEL, 0.02, sqrt(embed_var));
        mem_pool_reset(scratch);
    }

    /* 5c. Image_pos scale vs image features */
    if (vlm->image_pos && vlm->use_image_pos) {
        float *pd = tensor_data_ptr(vlm->image_pos);
        int np = tensor_numel(vlm->image_pos);
        tensor_stat ps = compute_stat(pd, np);
        printf("\n  [Image_pos scale]\n");
        printf("    image_pos RMS:   %.4f (init N(0,0.02))\n", ps.rms);
        printf("    init std=0.02 vs d_model=%d: pos can change features by ~%.1f%%\n",
               D_MODEL, 100.0 * 0.02 * sqrt(D_MODEL) / (1.0 * sqrt(D_MODEL)));
        printf("    >>> OK: 0.02 is 2%% of normalized features\n");
    }

    /* 5d. Residual branch scaling */
    {
        printf("\n  [Residual branch scaling (GPT-2 style)]\n");
        double base_std = 0.02;
        double residual_std_val = base_std / sqrt(2.0 * N_LAYERS);
        printf("    base_std=%.4f  residual_std=%.4f (=base_std/sqrt(2*L))\n",
               base_std, residual_std_val);
        printf("    d_model * base_std^2 = %d * %.4f^2 = %.4f\n",
               D_MODEL, base_std, (double)D_MODEL * base_std * base_std);
        printf("    >>> GPT-2 optimal: d_model*std^2 ≈ 0.3\n");
        printf("    >>> Here: %.4f — signals will shrink through residual stack\n",
               (double)D_MODEL * base_std * base_std);
        printf("    >>> For d_model=%d, optimal std ≈ sqrt(0.3/%d) = %.4f\n",
               D_MODEL, D_MODEL, sqrt(0.3 / D_MODEL));
    }

    /* 5e. Image_grad / embed_grad ratio */
    {
        printf("\n  [PATCH_LR_MULT=3.0 recommendation]\n");
        if (embed_grad_sq > 1e-30 && img_grad_sq > 1e-30) {
            double raw_ratio = sqrt(img_grad_sq) / sqrt(embed_grad_sq);
            printf("    Raw grad norm ratio (img/embed): %.3f\n", raw_ratio);
            printf("    Effective ratio with 3x mult:    %.3f\n", raw_ratio * 3.0);
            if (raw_ratio < 0.3)
                printf("    >>> Image grads much smaller than embed — 3x mult justified\n");
            else if (raw_ratio < 1.0)
                printf("    >>> Image grads moderately smaller — 3x mult reasonable\n");
            else if (raw_ratio < 3.0)
                printf("    >>> Image grads similar to embed — 3x mult may overshoot\n");
            else
                printf("    >>> Image grads larger than embed — 3x mult likely too high\n");
        }
    }

    mem_pool_reset(scratch);
    mem_pool_reset(data);
}

/* ══════════════════════════════════════════════════════════════════
 *  Main
 * ══════════════════════════════════════════════════════════════════ */

int main(void) {
    printf("╔══════════════════════════════════════════════════════════╗\n");
    printf("║     VLM Initialization Probe — Analytical Diagnostics   ║\n");
    printf("╚══════════════════════════════════════════════════════════╝\n");
    printf("Config: D=%d L=%d H=%d d_k=%d ff=%d | img=%dx%dx%d p%d\n",
           D_MODEL, N_LAYERS, N_HEADS, D_K, INTERMEDIATE,
           IMG_C, IMG_H, IMG_W, PATCH_SIZE);
    printf("Vocab=%d  Image tokens=%d  Max text T=%d\n\n",
           VOCAB_SIZE, N_IMG_TOK, TMAX);

    dnn_ctx ctx;
    dnn_ctx_init(&ctx, 256*1024*1024, (size_t)2*1024*1024*1024, 256*1024*1024);

    /* ── Create VLM (same config as imagenet_vlm.c) ── */
    vision_lm *vlm = vision_lm_create(ctx.params, VOCAB_SIZE, D_MODEL,
                                       N_LAYERS, N_HEADS, D_K, INTERMEDIATE,
                                       IMG_C, IMG_H, IMG_W, PATCH_SIZE, 1);
    vision_lm_init_weights(vlm);
    vision_lm_enable_rope(ctx.params, vlm, TMAX, 10000.0f);

    printf("Model created: n_params=%.2fM  n_img_tokens=%d\n\n",
           vision_lm_num_parameters(vlm) / 1e6, vlm->n_img_tokens);

    /* ── Run probe ── */
    run_probe(vlm, ctx.scratch, ctx.data);

    /* ── Cleanup ── */
    dnn_ctx_destroy(&ctx);
    printf("\n===== Probe complete =====\n");
    return 0;
}
