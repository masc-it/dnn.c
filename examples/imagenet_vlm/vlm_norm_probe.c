/* ══════════════════════════════════════════════════════════════════
 *  vlm_norm_probe v2 — compare norm placements for VLM gradient flow
 *
 *  Fixes from v1:
 *    - synthetic images use N(0,1) to match ImageNet norm
 *    - measures per-position RMS before/after final norm per modality
 *    - measures logit scale to explain loss difference
 *    - reports per-element grad std ratio (fair across param sizes)
 * ══════════════════════════════════════════════════════════════════ */

#include "dnn.h"
#include "gpt.h"
#include "vlm.h"
#include "optim.h"
#include "tokenizer.h"
#include "norm.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

#define IMG_H           224
#define IMG_W           224
#define IMG_C             3
#define PATCH_SIZE       16
#define D_MODEL         128
#define N_LAYERS          2
#define N_HEADS           2
#define D_K             64
#define INTERMEDIATE   256
#define VOCAB_SIZE      261
#define BATCH_SIZE       32
#define TMAX             48

static double rms_of(const float *d, int n) {
    double sum2 = 0;
    for (int i = 0; i < n; i++) sum2 += (double)d[i] * d[i];
    return sqrt(sum2 / n);
}

static double std_of(const float *d, int n) {
    double sum = 0, sum2 = 0;
    for (int i = 0; i < n; i++) { double v = d[i]; sum += v; sum2 += v*v; }
    return sqrt(sum2/n - (sum/n)*(sum/n));
}

static void zero_grads(module *m) {
    int n; tensor **ps = module_parameters(m, &n);
    for (int i = 0; i < n; i++) {
        float *g = tensor_grad(ps[i]);
        if (g) memset(g, 0, (size_t)tensor_numel(ps[i]) * sizeof(float));
    }
}

typedef struct { double grad_std; double grad_sq; } gstat;
static gstat gstat_of(tensor *p) {
    gstat g = {0,0};
    float *gd = tensor_grad(p);
    if (!gd) return g;
    int n = tensor_numel(p);
    double sq = 0;
    for (int i = 0; i < n; i++) sq += (double)gd[i] * gd[i];
    g.grad_sq = sq;
    g.grad_std = sqrt(sq / n);
    return g;
}

/* ── Config: which norms enabled ── */
typedef struct {
    const char *name;
    int use_img_norm;
    int use_text_norm;
} config_desc;

static void run_config(struct mem_pool *scratch, struct mem_pool *data,
                        vision_lm *vlm, rms_norm *text_norm,
                        const config_desc *desc,
                        const float *raw_img, const int *raw_inp,
                        const int *raw_tgt, const float *raw_msk,
                        const int *raw_tl) {
    int B = BATCH_SIZE, T = TMAX, I = vlm->n_img_tokens;
    int img_n = IMG_C * IMG_H * IMG_W * B;

    printf("─── %s ───\n", desc->name);

    /* ── Fresh tensors ── */
    tensor *images    = tensor_zeros_data(data, 4, (int[]){B, IMG_C, IMG_H, IMG_W});
    tensor *input_ids = tensor_zeros_data(data, 2, (int[]){B, T});
    tensor *target_ids= tensor_zeros_data(data, 2, (int[]){B, T});
    tensor *loss_mask = tensor_zeros_data(data, 2, (int[]){B, T});
    int tl[B];
    memcpy(tensor_data_ptr(images), raw_img, (size_t)img_n * sizeof(float));
    memcpy(tensor_data_ptr(input_ids), raw_inp, (size_t)B*T*sizeof(int));
    memcpy(tensor_data_ptr(target_ids), raw_tgt, (size_t)B*T*sizeof(int));
    memcpy(tensor_data_ptr(loss_mask), raw_msk, (size_t)B*T*sizeof(float));
    memcpy(tl, raw_tl, B*sizeof(int));

    zero_grads(&vlm->base);

    /* ── Build embeds ── */
    tensor *img_feat = vision_lm_image_embeds(scratch, vlm, images);
    if (desc->use_img_norm)
        img_feat = rms_norm_forward(scratch, vlm->image_norm, img_feat);
    if (vlm->use_image_pos && vlm->image_pos)
        img_feat = tensor_add(scratch, img_feat, vlm->image_pos);

    tensor *txt_feat = decoder_lm_token_embeds(scratch, vlm->lm, input_ids);
    if (desc->use_text_norm)
        txt_feat = rms_norm_forward(scratch, text_norm, txt_feat);

    tensor *embeds = tensor_cat(scratch, img_feat, txt_feat, 1);

    /* ── Concat scale ── */
    {
        tensor *ei = tensor_contiguous(scratch, tensor_slice(scratch, embeds, 1, 0, I));
        tensor *et = tensor_contiguous(scratch, tensor_slice(scratch, embeds, 1, I, T));
        double ir = rms_of(tensor_data_ptr(ei), tensor_numel(ei));
        double tr = rms_of(tensor_data_ptr(et), tensor_numel(et));
        printf("  Concat: img RMS=%.4f  txt RMS=%.4f  ratio=%.2f\n", ir, tr, ir/tr);
    }

    /* ── Forward through transformer blocks ── */
    tensor *h = embeds;
    for (int i = 0; i < vlm->lm->n_layers; i++)
        h = transformer_block_forward_ex(scratch, vlm->lm->blocks[i], h,
                                         ATTENTION_PREFIX_LM, I, NULL);

    /* ── Hidden stats BEFORE final norm (separate img/txt positions) ── */
    {
        tensor *hi = tensor_contiguous(scratch, tensor_slice(scratch, h, 1, 0, I));
        tensor *ht = tensor_contiguous(scratch, tensor_slice(scratch, h, 1, I, T));
        double h_img = rms_of(tensor_data_ptr(hi), tensor_numel(hi));
        double h_txt = rms_of(tensor_data_ptr(ht), tensor_numel(ht));
        printf("  Pre-norm h: img RMS=%.4f  txt RMS=%.4f\n", h_img, h_txt);
    }

    /* ── Final RMSNorm + lm_head ── */
    h = rms_norm_forward(scratch, vlm->lm->norm, h);
    {
        double hr = rms_of(tensor_data_ptr(h), tensor_numel(h));
        printf("  Post-norm h RMS: %.4f\n", hr);
    }

    tensor *logits = decoder_lm_lm_head_forward(scratch, vlm->lm, h);
    tensor *logits_text = tensor_contiguous(scratch, tensor_slice(scratch, logits, 1, I, T));

    /* ── Logit + softmax stats to explain CE gap ── */
    {
        float *ld = tensor_data_ptr(logits_text);
        int V = logits_text->shape[2];
        int BT = tensor_numel(logits_text) / V;  /* positions with grads */
        double m=0, v=0, lse_sum=0, target_logit_sum=0, n_tok=0;
        int min_lse_ix = 0; double max_lse = -1e30, min_lse = 1e30;
        for (int b = 0; b < logits_text->shape[0]; b++) {
            for (int t = 0; t < logits_text->shape[1]; t++) {
                int ti = t;
                float *row = ld + (long)b * logits_text->strides[0]
                                 + (long)ti * logits_text->strides[1];
                int target = ((int*)tensor_data_ptr(target_ids))[b * T + t];
                float mask_val = ((float*)tensor_data_ptr(loss_mask))[b * T + t];
                if (mask_val == 0.0f) continue;
                double mx = -INFINITY;
                for (int vv = 0; vv < V; vv++) if (row[vv] > mx) mx = row[vv];
                double se = 0;
                for (int vv = 0; vv < V; vv++) se += expf(row[vv] - mx);
                double lse = mx + log(se);
                lse_sum += lse;
                target_logit_sum += row[target];
                n_tok++;
                if (lse > max_lse) { max_lse = lse; }
                if (lse < min_lse) { min_lse = lse; }
                for (int vv = 0; vv < V; vv++) { double x = row[vv]; m += x; v += x*x; }
            }
        }
        m /= (n_tok * V); v = v/(n_tok*V) - m*m;
        double avg_lse = lse_sum / n_tok;
        double avg_tgt_logit = target_logit_sum / n_tok;
        double avg_ce = avg_lse - avg_tgt_logit;
        printf("  Logits: mean=%.4f  std=%.4f\n", m, sqrt(v));
        printf("  Per position: avg(logit_target)=%.2f  avg(logsumexp)=%.2f  avg(CE)=%.2f\n",
               avg_tgt_logit, avg_lse, avg_ce);
        printf("    logsumexp range: [%.2f, %.2f]  n_positions=%.0f\n", min_lse, max_lse, n_tok);
    }

    tensor *loss = tensor_cross_entropy_masked(scratch, logits_text,
                                               target_ids, loss_mask, 2);
    float lv = tensor_data_ptr(loss)[0];
    printf("  Loss: %.6f\n", lv);

    dnn_backward(scratch, loss);

    /* ── Grads per-param (per-element std, fair across sizes) ── */
    tensor *pw = module_find_param(&vlm->base, "patch_embed.weight");
    tensor *pb = module_find_param(&vlm->base, "patch_embed.bias");
    tensor *pn = module_find_param(&vlm->base, "image_norm.weight");
    tensor *pp = module_find_param(&vlm->base, "image_pos");
    tensor *ew = module_find_param(&vlm->base, "lm.embed.weight");
    tensor *lb = module_find_param(&vlm->base, "lm.lm_head.bias");

    gstat g_pw = gstat_of(pw);
    gstat g_ew = gstat_of(ew);
    gstat g_pb = gstat_of(pb);
    gstat g_pn = gstat_of(pn);
    gstat g_pp = gstat_of(pp);
    gstat g_lb = gstat_of(lb);

    printf("  Per-element grad std:\n");
    printf("    patch_embed.weight: %.4e  (%.0f params)\n", g_pw.grad_std, (double)tensor_numel(pw));
    printf("    lm.embed.weight:    %.4e  (%.0f params)\n", g_ew.grad_std, (double)tensor_numel(ew));
    printf("    patch_embed.bias:   %.4e\n", g_pb.grad_std);
    if (pn) printf("    image_norm.weight: %.4e\n", g_pn.grad_std);
    if (pp) printf("    image_pos:         %.4e\n", g_pp.grad_std);
    printf("    lm_head.bias:       %.4e\n", g_lb.grad_std);

    double ratio = (g_ew.grad_std > 1e-30) ? g_pw.grad_std / g_ew.grad_std : 0;
    printf("  Per-element grad ratio (patch/embed): %.4f", ratio);
    if      (ratio < 0.1)  printf("  <<< IMAGE DEAD\n");
    else if (ratio < 0.3)  printf("  image anemic\n");
    else if (ratio < 3.0)  printf("  BALANCED ✓\n");
    else                   printf("  image dominant\n");

    /* Params with grads: total + per-group */
    {
        double total_sq = 0, img_sq = 0, emb_sq = 0;
        int n_p; tensor **ps = module_parameters(&vlm->base, &n_p);
        for (int i = 0; i < n_p; i++) {
            float *g = tensor_grad(ps[i]);
            if (!g) continue;
            int nn = tensor_numel(ps[i]);
            double sq = 0; for (int j = 0; j < nn; j++) sq += (double)g[j] * g[j];
            total_sq += sq;
            if (ps[i] == pw || ps[i] == pb || ps[i] == pn || ps[i] == pp) img_sq += sq;
            else if (ps[i] == ew) emb_sq += sq;
        }
        printf("  Grad norm share: img=%.1f%%  embed=%.1f%%  other=%.1f%%\n",
               100.0*img_sq/total_sq, 100.0*emb_sq/total_sq,
               100.0*(total_sq-img_sq-emb_sq)/total_sq);
    }
    printf("\n");
}

int main(void) {
    printf("╔══════════════════════════════════════════════════════════╗\n");
    printf("║   VLM Norm Probe v2 — Gradient Flow Analysis           ║\n");
    printf("╚══════════════════════════════════════════════════════════╝\n");

    dnn_ctx ctx;
    dnn_ctx_init(&ctx, 256*1024*1024, (size_t)2*1024*1024*1024, 256*1024*1024);

    vision_lm *vlm = vision_lm_create(ctx.params, VOCAB_SIZE, D_MODEL,
                                       N_LAYERS, N_HEADS, D_K, INTERMEDIATE,
                                       IMG_C, IMG_H, IMG_W, PATCH_SIZE, 1);
    vision_lm_init_weights(vlm);
    vision_lm_enable_rope(ctx.params, vlm, TMAX, 10000.0f);

    rms_norm *text_norm = rms_norm_create(ctx.params, D_MODEL, 1e-5f);
    printf("Model: %lld params  D=%d L=%d\n\n", vision_lm_num_parameters(vlm), D_MODEL, N_LAYERS);

    /* ── Synthetic data: N(0,1) images (ImageNet norm), text tokens ── */
    int B = BATCH_SIZE, T = TMAX;
    int img_n = IMG_C * IMG_H * IMG_W * B;
    float *ri = (float*)malloc((size_t)img_n * sizeof(float));
    int   *rinp = (int*)malloc((size_t)B*T*sizeof(int));
    int   *rtgt = (int*)malloc((size_t)B*T*sizeof(int));
    float *rmsk = (float*)malloc((size_t)B*T*sizeof(float));
    int   *rtl  = (int*)malloc((size_t)B*sizeof(int));

    /* Box-Muller for N(0,1) images */
    for (int i = 0; i < img_n; i += 2) {
        float u1 = (float)rand()/RAND_MAX, u2 = (float)rand()/RAND_MAX;
        float r = sqrtf(-2*logf(u1+1e-10f));
        ri[i] = r * cosf(6.2831853f * u2);
        if (i+1 < img_n) ri[i+1] = r * sinf(6.2831853f * u2);
    }

    for (int b = 0; b < B; b++) {
        int len = (rand() % 20) + 5;
        if (len > T-2) len = T-2;
        for (int t = 0; t < T; t++) {
            rinp[b*T+t] = TOKENIZER_PAD_ID;
            rtgt[b*T+t] = TOKENIZER_PAD_ID;
            rmsk[b*T+t] = 0;
        }
        rinp[b*T] = TOKENIZER_BOS_ID;
        for (int t = 0; t < len; t++) {
            int tok = (rand() % 95) + 32;
            rinp[b*T+t+1] = tok;
            rtgt[b*T+t]   = tok;
            rmsk[b*T+t]   = 1.0f;
        }
        rtgt[b*T+len] = TOKENIZER_EOS_ID;
        rmsk[b*T+len] = 1.0f;
        rtl[b] = len + 1;
    }

    /* ── Run configs ── */
    config_desc cfgs[] = {
        {"A: no img_norm, no text_norm", 0, 0},
        {"B: img_norm,  no text_norm",   1, 0},
        {"C: img_norm,  text_norm",      1, 1},
        {"D: no img_norm, text_norm",    0, 1},
    };
    int nc = sizeof(cfgs)/sizeof(cfgs[0]);

    for (int ci = 0; ci < nc; ci++) {
        mem_pool_reset(ctx.scratch);
        mem_pool_reset(ctx.data);
        run_config(ctx.scratch, ctx.data, vlm, text_norm, &cfgs[ci],
                   ri, rinp, rtgt, rmsk, rtl);
    }

    /* ── Verify N(0,1) image stats ── */
    {
        printf("─── Image sanity check ───\n");
        double m=0, v=0;
        for (int i = 0; i < img_n; i++) { m += ri[i]; v += (double)ri[i]*ri[i]; }
        m /= img_n; v = v/img_n - m*m;
        printf("  Synth images: mean=%.4f  std=%.4f  (should be ~0, ~1)\n", m, sqrt(v));
    }

    free(ri); free(rinp); free(rtgt); free(rmsk); free(rtl);
    dnn_ctx_destroy(&ctx);
    printf("===== Done =====\n");
    return 0;
}
