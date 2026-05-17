/* ══════════════════════════════════════════════════════════════════
 *  vlm_debug_1 — diagnostic tool for VLM training issues
 *
 *  Tests:
 *    1. Gradient flow: check if image pathway gets meaningful gradients
 *    2. Image conditioning: compare loss with correct vs blank vs random image
 *    3. Per-sample loss distribution
 *    4. Autoregressive generation diagnostics
 *    5. Bucket sensitivity: same sample at different padding lengths
 * ══════════════════════════════════════════════════════════════════ */

#include "dnn.h"
#include "gpt.h"
#include "vlm.h"
#include "imagenet_vlm.h"
#include "optim.h"
#include "tokenizer.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include <sys/stat.h>
#include <sys/param.h>
#include <assert.h>

#define IMG_H           224
#define IMG_W           224
#define IMG_C             3
#define PATCH_SIZE       16
#define N_IMG_TOK       196
#define D_MODEL         128
#define VOCAB_SIZE      261
#define MAX_TEXT_LEN    128

/* ── Load checkpoint into existing vision_lm ── */
static int load_checkpoint(vision_lm *vlm, const char *path) {
    module_load(&vlm->base, path, 1);
    return 0;
}

/* ── Compute gradient norm of all params ── */
static float compute_grad_norm(tensor **params, int n_params) {
    double total = 0.0;
    for (int i = 0; i < n_params; i++) {
        float *g = tensor_grad(params[i]);
        if (!g) continue;
        int n = tensor_numel(params[i]);
        for (int j = 0; j < n; j++)
            total += (double)g[j] * (double)g[j];
    }
    return sqrtf((float)total);
}

/* ── Forward + backward + return loss + grad norm ── */
static float forward_and_backward(struct mem_pool *scratch, struct mem_pool *data,
                                   vision_lm *vlm, adamw_opt *opt,
                                   const tensor *images,
                                   const tensor *input_ids,
                                   const tensor *target_ids,
                                   const tensor *loss_mask,
                                   const int *text_lens,
                                   float *gnorm) {
    (void)data;
    int B = input_ids->shape[0];
    int Tmax = input_ids->shape[1];
    int I = vlm->n_img_tokens;

    int *combined_lens = _mem_pool_alloc(scratch, B * sizeof(int), NULL);
    for (int b = 0; b < B; b++) {
        int tl = text_lens[b];
        if (tl > Tmax) tl = Tmax;
        combined_lens[b] = I + tl;
    }

    adamw_zero_grad(opt);
    tensor *embeds = vision_lm_build_embeds(scratch, vlm, images, input_ids);
    tensor *logits = decoder_lm_forward_embeds_ex(scratch, vlm->lm, embeds,
                                                  ATTENTION_PREFIX_LM, I, combined_lens);
    tensor *logits_text = tensor_slice(scratch, logits, 1, I, Tmax);
    tensor *loss = tensor_cross_entropy_masked(scratch, logits_text, target_ids, loss_mask, 2);
    dnn_backward(scratch, loss);

    float gn = compute_grad_norm(opt->params, opt->n_params);
    if (gnorm) *gnorm = gn;
    return tensor_data_ptr(loss)[0];
}

/* ── Forward no-grad, return loss ── */
static float eval_loss(struct mem_pool *scratch, struct mem_pool *data,
                        vision_lm *vlm,
                        const tensor *images,
                        const tensor *input_ids,
                        const tensor *target_ids,
                        const tensor *loss_mask,
                        const int *text_lens) {
    (void)data;
    dnn_grad_ctx ng = dnn_no_grad_enter();

    int B = input_ids->shape[0];
    int Tmax = input_ids->shape[1];
    int I = vlm->n_img_tokens;

    int *combined_lens = _mem_pool_alloc(scratch, B * sizeof(int), NULL);
    for (int b = 0; b < B; b++) {
        int tl = text_lens[b];
        if (tl > Tmax) tl = Tmax;
        combined_lens[b] = I + tl;
    }

    tensor *embeds = vision_lm_build_embeds(scratch, vlm, images, input_ids);
    tensor *logits = decoder_lm_forward_embeds_ex(scratch, vlm->lm, embeds,
                                                  ATTENTION_PREFIX_LM, I, combined_lens);
    tensor *logits_text = tensor_slice(scratch, logits, 1, I, Tmax);
    tensor *loss = tensor_cross_entropy_masked(scratch, logits_text, target_ids, loss_mask, 2);

    float lv = tensor_data_ptr(loss)[0];
    dnn_no_grad_exit(ng);
    return lv;
}

/* ── Autoregressive generation (no cache) ── */
static int generate_text(struct mem_pool *scratch, struct mem_pool *data,
                          vision_lm *vlm,
                          const tensor *images,
                          int *output, int max_new) {
    dnn_grad_ctx ng = dnn_no_grad_enter();
    int I = vlm->n_img_tokens;
    int cur_len = 0;
    int *tokens = _mem_pool_alloc(data, (size_t)max_new * sizeof(int), NULL);
    tokens[cur_len++] = TOKENIZER_BOS_ID;

    for (int step = 1; step < max_new; step++) {
        tensor *prompt = tensor_zeros_data(data, 2, (int[]){1, cur_len});
        int *pd = (int *)prompt->data;
        memcpy(pd, tokens, (size_t)cur_len * sizeof(int));

        tensor *logits = vision_lm_forward(scratch, vlm, images, prompt);
        float *ld = tensor_data_ptr(logits);
        float *last_row = ld + (I + cur_len - 1) * vlm->vocab_size;

        int pred = 0;
        for (int v = 1; v < vlm->vocab_size; v++)
            if (last_row[v] > last_row[pred]) pred = v;

        tokens[cur_len++] = pred;
        if (pred == TOKENIZER_EOS_ID) break;
        mem_pool_reset(scratch);
    }

    memcpy(output, tokens, (size_t)cur_len * sizeof(int));
    dnn_no_grad_exit(ng);
    return cur_len;
}

/* ── Decode byte tokens to string ── */
static void decode_bytes(const int *tokens, int n, char *out, int max_out) {
    int pos = 0;
    for (int i = 0; i < n && pos < max_out - 1; i++) {
        int t = tokens[i];
        if (t == TOKENIZER_BOS_ID || t == TOKENIZER_PAD_ID) continue;
        if (t == TOKENIZER_EOS_ID) break;
        if (t >= 32 && t < 127)
            out[pos++] = (char)t;
        else
            out[pos++] = '?';
    }
    out[pos] = '\0';
}

/* ── Copy a single sample (index s) from batch tensors ── */
static void copy_sample(tensor *dst_img, tensor *dst_inp, tensor *dst_tgt, tensor *dst_msk,
                         const tensor *src_img, const tensor *src_inp,
                         const tensor *src_tgt, const tensor *src_msk,
                         int s, int T) {
    float *simg = tensor_data_ptr((tensor *)src_img);
    float *dimg = tensor_data_ptr(dst_img);
    int npx = IMG_C * IMG_H * IMG_W;
    memcpy(dimg, simg + (long)s * npx, (size_t)npx * sizeof(float));

    int *sinp = (int *)src_inp->data;
    int *dinp = (int *)dst_inp->data;
    memcpy(dinp, sinp + (long)s * T, (size_t)T * sizeof(int));

    int *stgt = (int *)src_tgt->data;
    int *dtgt = (int *)dst_tgt->data;
    memcpy(dtgt, stgt + (long)s * T, (size_t)T * sizeof(int));

    float *smsk = tensor_data_ptr((tensor *)src_msk);
    float *dmsk = tensor_data_ptr(dst_msk);
    memcpy(dmsk, smsk + (long)s * T, (size_t)T * sizeof(float));
}

/* ── Find param by path substring in module tree ── */
static int find_param_by_substr(tensor **params, int n_params,
                                 const char *substr, int *idxs, int max_idxs) {
    int found = 0;
    for (int i = 0; i < n_params && found < max_idxs; i++) {
        /* Use module_find_param — check if we can identify by known path patterns */
        /* Since we can't access name from tensor, we need to use heuristics */
        /* For now, just report total grad norm and let user interpret */
        (void)params; (void)substr; (void)idxs; (void)max_idxs;
    }
    return found;
}

/* ── Per-param-group norm using module_find_param ── */
static void report_grad_by_group(vision_lm *vlm, tensor **params, int n_params) {
    const char *img_params[] = {
        "patch_embed.weight", "patch_embed.bias",
        "image_norm.weight",
        "image_pos",
        NULL
    };

    double total_sq = 0.0, img_sq = 0.0, embed_sq = 0.0, lm_sq = 0.0;

    /* Sum all grad norms */
    for (int i = 0; i < n_params; i++) {
        float *g = tensor_grad(params[i]);
        if (!g) continue;
        int n = tensor_numel(params[i]);
        double nsq = 0.0;
        for (int j = 0; j < n; j++) nsq += (double)g[j] * (double)g[j];
        total_sq += nsq;
    }

    /* Image pathway params via module_find_param */
    for (int pi = 0; img_params[pi]; pi++) {
        tensor *t = module_find_param(&vlm->base, img_params[pi]);
        if (t && tensor_grad(t)) {
            float *g = tensor_grad(t);
            int n = tensor_numel(t);
            double nsq = 0.0;
            for (int j = 0; j < n; j++) nsq += (double)g[j] * (double)g[j];
            img_sq += nsq;
        }
    }

    /* Embedding table */
    tensor *emb = module_find_param(&vlm->base, "lm.embed.weight");
    if (emb && tensor_grad(emb)) {
        float *g = tensor_grad(emb);
        int n = tensor_numel(emb);
        double nsq = 0.0;
        for (int j = 0; j < n; j++) nsq += (double)g[j] * (double)g[j];
        embed_sq += nsq;
    }

    lm_sq = total_sq - img_sq - embed_sq;
    if (lm_sq < 0) lm_sq = 0;

    printf("    Image-path grad norm:   %.4e  (%.1f%%)\n",
           sqrt((float)img_sq), 100.0 * img_sq / (total_sq + 1e-30));
    printf("    Embedding grad norm:    %.4e  (%.1f%%)\n",
           sqrt((float)embed_sq), 100.0 * embed_sq / (total_sq + 1e-30));
    printf("    LM-path grad norm:      %.4e  (%.1f%%)\n",
           sqrt((float)lm_sq), 100.0 * lm_sq / (total_sq + 1e-30));
    printf("    Total grad norm:        %.4e\n", sqrt((float)total_sq));
}

/* ══════════════════════════════════════════════════════════════════
 *  Main
 * ══════════════════════════════════════════════════════════════════ */

int main(int argc, char **argv) {
    printf("=== VLM Debug v1 ===\n\n");

    const char *data_dir = NULL;
    const char *ckpt_path = NULL;
    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "--data-dir") == 0 && i + 1 < argc)
            data_dir = argv[++i];
        else if (strcmp(argv[i], "--ckpt") == 0 && i + 1 < argc)
            ckpt_path = argv[++i];
        else if (strcmp(argv[i], "--help") == 0 || strcmp(argv[i], "-h") == 0) {
            printf("Usage: %s --data-dir PATH [--ckpt PATH]\n", argv[0]);
            return 0;
        }
    }
    if (!data_dir) {
        fprintf(stderr, "Error: --data-dir PATH required\n");
        return 1;
    }

    /* Find latest checkpoint */
    if (!ckpt_path) {
        FILE *ls = popen("ls -t ckpt/imagenet_vlm_epoch*.bin 2>/dev/null | head -1", "r");
        if (ls) {
            char buf[256];
            if (fgets(buf, sizeof(buf), ls)) {
                size_t len = strlen(buf);
                if (len > 0 && buf[len-1] == '\n') buf[len-1] = '\0';
                ckpt_path = strdup(buf);
            }
            pclose(ls);
        }
    }
    if (!ckpt_path) {
        fprintf(stderr, "No checkpoint found. Use --ckpt PATH\n");
        return 1;
    }
    printf("Checkpoint: %s\n", ckpt_path);

    /* ── Pools ── */
    dnn_ctx ctx;
    dnn_ctx_init(&ctx, 256*1024*1024, (size_t)2*1024*1024*1024, 256*1024*1024);

    /* ── Create VLM ── */
    srand(42);
    vision_lm *vlm = vision_lm_create(ctx.params, VOCAB_SIZE, D_MODEL,
                                       2, 2, 64, 256,
                                       IMG_C, IMG_H, IMG_W, PATCH_SIZE, 1);
    vision_lm_init_weights(vlm);
    vision_lm_enable_rope(ctx.params, vlm, MAX_TEXT_LEN, 10000.0f);

    if (load_checkpoint(vlm, ckpt_path) != 0) {
        dnn_ctx_destroy(&ctx);
        return 1;
    }
    printf("Model: %.2fM params\n", vision_lm_num_parameters(vlm) / 1e6);

    /* ── Load labels ── */
    char labels_path[MAXPATHLEN];
    snprintf(labels_path, sizeof(labels_path), "%s/labels.txt", data_dir);
    FILE *lf = fopen(labels_path, "r");
    char *label_names[1000];
    int n_labels = 0;
    if (lf) {
        char buf[256];
        while (fgets(buf, sizeof(buf), lf) && n_labels < 1000) {
            size_t len = strlen(buf);
            if (len > 0 && buf[len-1] == '\n') buf[len-1] = '\0';
            label_names[n_labels] = strdup(buf);
            n_labels++;
        }
        fclose(lf);
    }
    printf("Labels: %d\n", n_labels);

    /* ── Val dataloader ── */
    imagenet_vlm_dl *val_dl = imagenet_vlm_dl_create("val", data_dir, 0, 999);
    if (!val_dl) { fprintf(stderr, "Failed val dl\n"); return 1; }
    printf("Val samples: %d\n\n", imagenet_vlm_dl_total(val_dl));

    int n_params;
    tensor **all_params = module_parameters(&vlm->base, &n_params);

    int N_BUCKETS = 4;
    int bucket_limits[] = {33, 65, 97, 129};
    int bucket_starts[5];

    /* ═══════════════════════════════════════════════════════════════
     *  TEST 1: Image conditioning
     * ═══════════════════════════════════════════════════════════════ */
    printf("═══ Test 1: Image conditioning ═══\n");

    int test_bs = 8;
    int T_test = 32;

    tensor *img_orig = tensor_zeros_data(ctx.data, 4, (int[]){test_bs, IMG_C, IMG_H, IMG_W});
    tensor *inp_ids  = tensor_zeros_data(ctx.data, 2, (int[]){test_bs, T_test});
    tensor *tgt_ids  = tensor_zeros_data(ctx.data, 2, (int[]){test_bs, T_test});
    tensor *loss_msk = tensor_scratch(ctx.scratch, 2, (int[]){test_bs, T_test}, 0);
    int text_lens[test_bs], label_ids[test_bs];

    imagenet_vlm_dl_reset(val_dl);
    int got = imagenet_vlm_dl_next_batch(val_dl, img_orig, inp_ids, tgt_ids,
                                          loss_msk, text_lens, label_ids, test_bs);
    if (got <= 0) return 1;
    printf("Got %d samples\n", got);

    /* A. Correct image */
    float loss_correct = eval_loss(ctx.scratch, ctx.data, vlm,
                                   img_orig, inp_ids, tgt_ids, loss_msk, text_lens);
    printf("  Loss (correct img):         %.6f\n", loss_correct);

    /* B. Blank image (all zeros = mean pixel) */
    tensor *img_blank = tensor_zeros_data(ctx.data, 4, (int[]){test_bs, IMG_C, IMG_H, IMG_W});
    float loss_blank = eval_loss(ctx.scratch, ctx.data, vlm,
                                 img_blank, inp_ids, tgt_ids, loss_msk, text_lens);
    printf("  Loss (blank img):           %.6f\n", loss_blank);

    /* C. Random noise image */
    tensor *img_noise = tensor_zeros_data(ctx.data, 4, (int[]){test_bs, IMG_C, IMG_H, IMG_W});
    {
        float *nd = tensor_data_ptr(img_noise);
        int nn = tensor_numel(img_noise);
        for (int i = 0; i < nn; i++) nd[i] = (float)rand() / (float)RAND_MAX * 4.0f - 2.0f;
    }
    float loss_noise = eval_loss(ctx.scratch, ctx.data, vlm,
                                 img_noise, inp_ids, tgt_ids, loss_msk, text_lens);
    printf("  Loss (random noise img):    %.6f\n", loss_noise);

    /* D. Swapped images (wrong image for each label) */
    tensor *img_swapped = tensor_zeros_data(ctx.data, 4, (int[]){test_bs, IMG_C, IMG_H, IMG_W});
    {
        float *src = tensor_data_ptr(img_orig);
        float *dst = tensor_data_ptr(img_swapped);
        int sample_bytes = IMG_C * IMG_H * IMG_W;
        if (got >= 2) {
            memcpy(dst, src + 1 * sample_bytes, (size_t)sample_bytes * sizeof(float));
            memcpy(dst + 1 * sample_bytes, src, (size_t)sample_bytes * sizeof(float));
        }
        for (int i = 2; i < got; i++)
            memcpy(dst + i * sample_bytes, src + i * sample_bytes,
                   (size_t)sample_bytes * sizeof(float));
    }
    float loss_swapped = eval_loss(ctx.scratch, ctx.data, vlm,
                                   img_swapped, inp_ids, tgt_ids, loss_msk, text_lens);
    printf("  Loss (swapped img):         %.6f\n", loss_swapped);

    float cond_gap = loss_blank - loss_noise;
    float cond_gap2 = loss_correct - loss_noise;
    printf("  Conditioning gap (blank - noise):  %+.6f\n", cond_gap);
    printf("  Conditioning gap (correct - noise): %+.6f\n", cond_gap2);
    if (cond_gap2 < -0.1)
        printf("  >>> WARNING: model prefers noise over correct images!\n");
    else if (cond_gap2 < 0.01)
        printf("  >>> WARNING: tiny gap — model effectively ignores images\n");
    else if (cond_gap2 < 0.5)
        printf("  >>> NOTE: weak conditioning — gap exists but small\n");
    else
        printf("  >>> OK: healthy conditioning gap\n");

    mem_pool_reset(ctx.scratch); /* no data reset, keep r_* alive */
    mem_pool_reset(ctx.data);

    /* ═══════════════════════════════════════════════════════════════
     *  TEST 2: Gradient flow per parameter group
     * ═══════════════════════════════════════════════════════════════ */
    printf("\n═══ Test 2: Gradient flow ═══\n");

    /* Re-create batch */
    img_orig = tensor_zeros_data(ctx.data, 4, (int[]){test_bs, IMG_C, IMG_H, IMG_W});
    inp_ids  = tensor_zeros_data(ctx.data, 2, (int[]){test_bs, T_test});
    tgt_ids  = tensor_zeros_data(ctx.data, 2, (int[]){test_bs, T_test});
    loss_msk = tensor_scratch(ctx.scratch, 2, (int[]){test_bs, T_test}, 0);

    imagenet_vlm_dl_reset(val_dl);
    got = imagenet_vlm_dl_next_batch(val_dl, img_orig, inp_ids, tgt_ids,
                                      loss_msk, text_lens, label_ids, test_bs);
    if (got <= 0) return 1;

    /* Create temporary optimizer just for params list + zero_grad */
    adamw_opt *tmp_opt = adamw_create(ctx.params, all_params, n_params,
                                       0.0f, 0.9f, 0.999f, 1e-8f, 1e-4f);

    float gn;
    float loss_grad = forward_and_backward(ctx.scratch, ctx.data, vlm,
                                            tmp_opt,
                                            img_orig, inp_ids, tgt_ids, loss_msk,
                                            text_lens, &gn);
    printf("  Loss (with backward): %.6f\n", loss_grad);
    report_grad_by_group(vlm, all_params, n_params);

    mem_pool_reset(ctx.scratch); /* no data reset, keep r_* alive */
    mem_pool_reset(ctx.data);

    /* ═══════════════════════════════════════════════════════════════
     *  TEST 3: Per-sample loss — is the model actually using images?
     * ═══════════════════════════════════════════════════════════════ */
    printf("\n═══ Test 3: Per-sample loss (comparing correct vs noise image) ═══\n");

    imagenet_vlm_dl_reset(val_dl);
    {
        int ps_bs = 1;
        int T_ps = 32;

        for (int s = 0; s < 8; s++) {
            tensor *img_s = tensor_zeros_data(ctx.data, 4, (int[]){ps_bs, IMG_C, IMG_H, IMG_W});
            tensor *inp_s = tensor_zeros_data(ctx.data, 2, (int[]){ps_bs, T_ps});
            tensor *tgt_s = tensor_zeros_data(ctx.data, 2, (int[]){ps_bs, T_ps});
            tensor *msk_s = tensor_scratch(ctx.scratch, 2, (int[]){ps_bs, T_ps}, 0);
            int tl_s[1];

            int *lbl_s = _mem_pool_alloc(ctx.data, sizeof(int), NULL);
            /* Get next sample from dataloader (single sample batch) */
            int pg = imagenet_vlm_dl_next_batch(val_dl, img_s, inp_s, tgt_s,
                                                  msk_s, tl_s, lbl_s, ps_bs);
            if (pg <= 0) break;

            float ls = eval_loss(ctx.scratch, ctx.data, vlm,
                                 img_s, inp_s, tgt_s, msk_s, tl_s);

            /* Noise image loss for same sample */
            tensor *noise_s = tensor_zeros_data(ctx.data, 4, (int[]){ps_bs, IMG_C, IMG_H, IMG_W});
            {
                float *nd = tensor_data_ptr(noise_s);
                int nn = tensor_numel(noise_s);
                for (int i = 0; i < nn; i++)
                    nd[i] = (float)rand() / (float)RAND_MAX * 4.0f - 2.0f;
            }
            float ls_noise = eval_loss(ctx.scratch, ctx.data, vlm,
                                       noise_s, inp_s, tgt_s, msk_s, tl_s);

            char label_str[256] = "?";
            if (lbl_s[0] >= 0 && lbl_s[0] < n_labels)
                snprintf(label_str, sizeof(label_str), "%s", label_names[lbl_s[0]]);

            printf("  [%d] class=%-3d label=%-30s loss=%.4f  noise-loss=%.4f  gap=%+.4f\n",
                   s, lbl_s[0], label_str, ls, ls_noise, ls - ls_noise);

            mem_pool_reset(ctx.scratch); /* no data reset, keep r_* alive */
            mem_pool_reset(ctx.data);
        }
    }

    /* ═══════════════════════════════════════════════════════════════
     *  TEST 4: Autoregressive generation
     * ═══════════════════════════════════════════════════════════════ */
    printf("\n═══ Test 4: Generation diagnostics ═══\n");

    imagenet_vlm_dl_reset(val_dl);
    {
        int gen_bs = 5;
        int T_gen = 32;

        tensor *g_img = tensor_zeros_data(ctx.data, 4, (int[]){gen_bs, IMG_C, IMG_H, IMG_W});
        tensor *g_inp = tensor_zeros_data(ctx.data, 2, (int[]){gen_bs, T_gen});
        tensor *g_tgt = tensor_zeros_data(ctx.data, 2, (int[]){gen_bs, T_gen});
        tensor *g_msk = tensor_scratch(ctx.scratch, 2, (int[]){gen_bs, T_gen}, 0);
        int g_tl[gen_bs], g_lbl[gen_bs];

        int gg = imagenet_vlm_dl_next_batch(val_dl, g_img, g_inp, g_tgt,
                                             g_msk, g_tl, g_lbl, gen_bs);
        if (gg > 0) {
            for (int s = 0; s < gg; s++) {
                tensor *img1 = tensor_zeros_data(ctx.data, 4, (int[]){1, IMG_C, IMG_H, IMG_W});
                float *src = tensor_data_ptr(g_img);
                float *dst = tensor_data_ptr(img1);
                memcpy(dst, src + (long)s * IMG_C * IMG_H * IMG_W,
                       (size_t)(IMG_C * IMG_H * IMG_W) * sizeof(float));

                /* Also test with blank image for comparison */
                tensor *img1_blank = tensor_zeros_data(ctx.data, 4, (int[]){1, IMG_C, IMG_H, IMG_W});

                int output[128], output_blank[128];
                int n_out = generate_text(ctx.scratch, ctx.data, vlm,
                                          img1, output, 128);
                int n_out_blank = generate_text(ctx.scratch, ctx.data, vlm,
                                                img1_blank, output_blank, 128);

                char predicted[256], predicted_blank[256];
                decode_bytes(output, n_out, predicted, sizeof(predicted));
                decode_bytes(output_blank, n_out_blank, predicted_blank, sizeof(predicted_blank));

                char label_str[256] = "?";
                if (g_lbl[s] >= 0 && g_lbl[s] < n_labels)
                    snprintf(label_str, sizeof(label_str), "%s", label_names[g_lbl[s]]);

                printf("  [%d] true=%-30s\n"
                       "        pred=              %-30s\n"
                       "        pred(blank)=       %-30s\n",
                       g_lbl[s], label_str, predicted, predicted_blank);

                mem_pool_reset(ctx.scratch); /* no data reset, keep r_* alive */
                mem_pool_reset(ctx.data);
            }
        }
    }

    /* ═══════════════════════════════════════════════════════════════
     *  TEST 5: Bucket sensitivity
     * ═══════════════════════════════════════════════════════════════ */
    printf("\n═══ Test 5: Same sample at different padding lengths ═══\n");

    imagenet_vlm_dl_reset(val_dl);
    {
        int T_lens[] = {32, 64, 96, 128};
        const char *b_names[] = {"T=32", "T=64", "T=96", "T=128"};

        /* Get one raw sample */
        int rbs = 1;
        tensor *r_img = tensor_zeros_data(ctx.data, 4, (int[]){rbs, IMG_C, IMG_H, IMG_W});
        tensor *r_inp = tensor_zeros_data(ctx.data, 2, (int[]){rbs, 128});
        tensor *r_tgt = tensor_zeros_data(ctx.data, 2, (int[]){rbs, 128});
        tensor *r_msk = tensor_scratch(ctx.scratch, 2, (int[]){rbs, 128}, 0);
        int r_tl[1], r_lbl[1];

        int rg = imagenet_vlm_dl_next_batch(val_dl, r_img, r_inp, r_tgt, r_msk,
                                             r_tl, r_lbl, rbs);
        if (rg > 0) {
            int full_len = r_tl[0];  /* BOS + text + EOS */
            int stored_len = full_len - 1;

            /* Save raw data to stack so pool resets don't invalidate */
            float raw_img[IMG_C * IMG_H * IMG_W];
            int raw_inp[128], raw_tgt[128];
            float raw_msk[128];
            memcpy(raw_img, tensor_data_ptr(r_img), sizeof(raw_img));
            memcpy(raw_inp, r_inp->data, (size_t)full_len * sizeof(int));
            memcpy(raw_tgt, r_tgt->data, (size_t)stored_len * sizeof(int));
            memcpy(raw_msk, tensor_data_ptr(r_msk), 128 * sizeof(float));

            for (int bni = 0; bni < 4; bni++) {
                int Tb = T_lens[bni];

                if (stored_len + 1 > Tb) {
                    printf("  %s  stored_len=%d  SKIP (exceeds T)\n",
                           b_names[bni], stored_len);
                    continue;
                }

                tensor *b_img = tensor_zeros_data(ctx.data, 4, (int[]){rbs, IMG_C, IMG_H, IMG_W});
                tensor *b_inp = tensor_zeros_data(ctx.data, 2, (int[]){rbs, Tb});
                tensor *b_tgt = tensor_zeros_data(ctx.data, 2, (int[]){rbs, Tb});
                tensor *b_msk = tensor_scratch(ctx.scratch, 2, (int[]){rbs, Tb}, 0);

                /* Copy image from saved stack buffer */
                memcpy(tensor_data_ptr(b_img), raw_img,
                       (size_t)(IMG_C * IMG_H * IMG_W) * sizeof(float));

                /* Build input_ids: [BOS, byte0, ..., EOS, PAD...] */
                int *bip = (int *)b_inp->data;
                memcpy(bip, raw_inp, (size_t)full_len * sizeof(int));
                for (int t = full_len; t < Tb; t++) bip[t] = IMAGENET_PAD_ID;

                /* Build target_ids: [byte0, ..., EOS, PAD...] */
                int *btp = (int *)b_tgt->data;
                memcpy(btp, raw_tgt, (size_t)stored_len * sizeof(int));
                for (int t = stored_len; t < Tb; t++) btp[t] = IMAGENET_PAD_ID;

                /* Build loss_mask: 1 for stored_len positions */
                float *bm = tensor_data_ptr(b_msk);
                for (int t = 0; t < stored_len; t++) bm[t] = 1.0f;
                for (int t = stored_len; t < Tb; t++) bm[t] = 0.0f;

                int tl_b[1] = {full_len};
                float loss_b = eval_loss(ctx.scratch, ctx.data, vlm,
                                         b_img, b_inp, b_tgt, b_msk, tl_b);

                printf("  %s  full_len=%d  stored=%d  loss=%.6f\n",
                       b_names[bni], full_len, stored_len, loss_b);

                mem_pool_reset(ctx.scratch); /* no data reset, keep r_* alive */
                mem_pool_reset(ctx.data);
            }
        }
    }

    /* ═══════════════════════════════════════════════════════════════
     *  Cleanup
     * ═══════════════════════════════════════════════════════════════ */
    imagenet_vlm_dl_free(val_dl);
    if (label_names) {
        for (int i = 0; i < n_labels; i++) free(label_names[i]);
    }
    dnn_ctx_destroy(&ctx);
    printf("\n===== Done =====\n");
    return 0;
}
