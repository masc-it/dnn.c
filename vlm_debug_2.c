/* ══════════════════════════════════════════════════════════════════
 *  vlm_debug_2 — Diagnose VLM training issues
 *
 *  Loads checkpoint, runs controlled forward passes,
 *  measures attention patterns, per-layer gradients,
 *  image influence, and bucket crossover behavior.
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

/* ── Config (must match training) ── */
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
#define GRAD_CLIP        5.0f
#define I               N_IMG_TOK

/* ── Helper: row statistics ── */
static float row_max_val(const float *row, int V) {
    float mx = -INFINITY;
    for (int v = 0; v < V; v++) if (row[v] > mx) mx = row[v];
    return mx;
}

static float row_softmax_denom(const float *row, int V, float mx) {
    float se = 0.0f;
    for (int v = 0; v < V; v++) se += expf(row[v] - mx);
    return se;
}

/* ── Per-layer grad norm helper ── */
typedef struct {
    const char *name;
    tensor     *param;
} named_param;

static int collect_named_params(module *m, named_param *list, int max, int *n) {
    if (*n >= max) return 0;
    module_item *item = m->items_head;
    while (item && *n < max) {
        if (item->kind == MODULE_ITEM_PARAM) {
            list[*n].name  = item->name;
            list[*n].param = item->as.param;
            (*n)++;
        } else if (item->kind == MODULE_ITEM_CHILD) {
            collect_named_params(item->as.child, list, max, n);
        }
        item = item->next;
    }
    return 0;
}

static void print_grad_norms(module *m, const char *heading) {
    named_param list[256];
    int n = 0;
    collect_named_params(m, list, 256, &n);
    printf("  ── %s (%d params) ──\n", heading, n);
    double total = 0.0;
    for (int i = 0; i < n; i++) {
        float *g = tensor_grad(list[i].param);
        if (!g) continue;
        int numel = tensor_numel(list[i].param);
        double nsq = 0.0;
        for (int j = 0; j < numel; j++) nsq += (double)g[j] * (double)g[j];
        float gn = sqrtf((float)nsq);
        if (gn > 1e-6f) {
            printf("    %-30s  shape=[", list[i].name);
            for (int d = 0; d < list[i].param->ndim; d++)
                printf("%s%d", d ? "," : "", list[i].param->shape[d]);
            printf("]  gn=%.4e\n", gn);
        }
        total += nsq;
    }
    printf("  total gn=%.4e\n", sqrtf((float)total));
}

/* ══════════════════════════════════════════════════════════════════
 *  TEST: Image influence — compare logits from real vs blank image
 * ══════════════════════════════════════════════════════════════════ */

static void test_image_influence(struct mem_pool *scratch, struct mem_pool *data,
                                  vision_lm *vlm, const tensor *real_img,
                                  const tensor *blank_img, const tensor *text_ids) {
    (void)data;
    int B = text_ids->shape[0];
    int T = text_ids->shape[1];
    int V = vlm->vocab_size;

    dnn_grad_ctx ng = dnn_no_grad_enter();

    tensor *logits_real = vision_lm_forward(scratch, vlm, real_img, text_ids);
    float *lr = tensor_data_ptr(logits_real);
    tensor *logits_blank = vision_lm_forward(scratch, vlm, blank_img, text_ids);
    float *lb = tensor_data_ptr(logits_blank);

    float max_diff = 0.0f;
    double kl_div = 0.0;
    int n_text_pos = 0;

    for (int b = 0; b < B; b++) {
        for (int t = I; t < I + T; t++) {
            n_text_pos++;
            float *r_row = lr + b * logits_real->strides[0] + t * V;
            float *b_row = lb + b * logits_blank->strides[0] + t * V;
            for (int v = 0; v < V; v++) {
                float d = fabsf(r_row[v] - b_row[v]);
                if (d > max_diff) max_diff = d;
            }
            /* Softmax */
            float mx_r = row_max_val(r_row, V);
            float mx_b = row_max_val(b_row, V);
            float se_r = row_softmax_denom(r_row, V, mx_r);
            float se_b = row_softmax_denom(b_row, V, mx_b);
            float inv_se_r = 1.0f / se_r;
            float inv_se_b = 1.0f / se_b;
            for (int v = 0; v < V; v++) {
                float pr = expf(r_row[v] - mx_r) * inv_se_r;
                float pb = expf(b_row[v] - mx_b) * inv_se_b;
                if (pr > 1e-10f && pb > 1e-10f)
                    kl_div += pr * logf(pr / pb);
            }
        }
    }
    printf("  max_logit_diff=%.4e  kl_div=%.4e  n_positions=%d\n",
           max_diff, kl_div, n_text_pos);

    /* Argmax agreement at first text position */
    int argmax_r = 0, argmax_b = 0;
    float *first_r = lr + 0 * logits_real->strides[0] + I * V;
    float *first_b = lb + 0 * logits_blank->strides[0] + I * V;
    for (int v = 1; v < V; v++) {
        if (first_r[v] > first_r[argmax_r]) argmax_r = v;
        if (first_b[v] > first_b[argmax_b]) argmax_b = v;
    }

    char ar_str[16], ab_str[16];
    if (argmax_r < 256) { ar_str[0] = (char)argmax_r; ar_str[1] = '\0'; }
    else if (argmax_r == 257) strcpy(ar_str, "BOS");
    else if (argmax_r == 258) strcpy(ar_str, "EOS");
    else snprintf(ar_str, sizeof(ar_str), "%d", argmax_r);
    if (argmax_b < 256) { ab_str[0] = (char)argmax_b; ab_str[1] = '\0'; }
    else if (argmax_b == 257) strcpy(ab_str, "BOS");
    else if (argmax_b == 258) strcpy(ab_str, "EOS");
    else snprintf(ab_str, sizeof(ab_str), "%d", argmax_b);

    printf("  argmax@text[0]: real=%s  blank=%s  %s\n",
           ar_str, ab_str, argmax_r == argmax_b ? "SAME (BAD)" : "DIFFERENT (GOOD)");

    /* P(EOS) at first text position */
    float mx_r = row_max_val(first_r, V);
    float se_r = row_softmax_denom(first_r, V, mx_r);
    float mx_b = row_max_val(first_b, V);
    float se_b = row_softmax_denom(first_b, V, mx_b);
    float p_eos_r = expf(first_r[TOKENIZER_EOS_ID] - mx_r) / se_r;
    float p_eos_b = expf(first_b[TOKENIZER_EOS_ID] - mx_b) / se_b;
    printf("  P(EOS)@text[0]: real=%.4e  blank=%.4e\n", p_eos_r, p_eos_b);

    mem_pool_reset(scratch);
    dnn_no_grad_exit(ng);
}

/* ══════════════════════════════════════════════════════════════════
 *  Forward-only loss (for bucket crossover test)
 * ══════════════════════════════════════════════════════════════════ */

static float forward_loss(struct mem_pool *scratch, struct mem_pool *data,
                           vision_lm *vlm, const tensor *img,
                           const tensor *inp, const tensor *tgt,
                           const tensor *msk, const int *tl, int B, int Tmax) {
    (void)data;
    dnn_grad_ctx ng = dnn_no_grad_enter();
    int *combined_lens = _mem_pool_alloc(scratch, B * sizeof(int), NULL);
    for (int b = 0; b < B; b++) {
        int len = tl[b] > Tmax ? Tmax : tl[b];
        combined_lens[b] = I + len;
    }
    tensor *embeds = vision_lm_build_embeds(scratch, vlm, img, inp);
    tensor *logits = decoder_lm_forward_embeds_ex(scratch, vlm->lm, embeds,
                                                  ATTENTION_PREFIX_LM, I, combined_lens);
    tensor *lt = tensor_slice(scratch, logits, 1, I, Tmax);
    tensor *loss = tensor_cross_entropy_masked(scratch, lt, tgt, msk, 2);
    float val = tensor_data_ptr(loss)[0];
    mem_pool_reset(scratch);
    dnn_no_grad_exit(ng);
    return val;
}

/* ══════════════════════════════════════════════════════════════════
 *  Main
 * ══════════════════════════════════════════════════════════════════ */

int main(int argc, char **argv) {
    printf("=== VLM Debug v2 ===\n\n");

    if (argc < 3) {
        fprintf(stderr, "Usage: %s --data-dir PATH [--ckpt PATH]\n", argv[0]);
        return 1;
    }

    const char *data_dir = NULL;
    const char *ckpt_path = NULL;
    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "--data-dir") == 0 && i+1 < argc) data_dir = argv[++i];
        else if (strcmp(argv[i], "--ckpt") == 0 && i+1 < argc) ckpt_path = argv[++i];
    }
    if (!data_dir) { fprintf(stderr, "Error: --data-dir required\n"); return 1; }

    /* ── Pools ── */
    dnn_ctx ctx;
    dnn_ctx_init(&ctx, 256*1024*1024, (size_t)4096*1024*1024, 256*1024*1024);
    srand(42);

    /* ── Create VLM ── */
    vision_lm *vlm = vision_lm_create(ctx.params, VOCAB_SIZE, D_MODEL,
                                       N_LAYERS, N_HEADS, D_K, INTERMEDIATE,
                                       IMG_C, IMG_H, IMG_W, PATCH_SIZE, 1);
    vision_lm_init_weights(vlm);
    vision_lm_enable_rope(ctx.params, vlm, 256, 10000.0f);

    if (ckpt_path) {
        printf("Loading checkpoint: %s\n", ckpt_path);
        module_load(&vlm->base, ckpt_path, 1);
    } else {
        printf("WARNING: No checkpoint loaded.  Random init.\n");
    }

    /* ── Dataloader ── */
    imagenet_vlm_dl *dl = imagenet_vlm_dl_create("train", data_dir, 0, 42);
    if (!dl) { fprintf(stderr, "Failed to create dataloader\n"); return 1; }
    printf("Dataloader: %d samples, %d shards\n", dl->total_samples, dl->num_shards);

    /* ── Load label names ── */
    char labels_path[MAXPATHLEN];
    snprintf(labels_path, sizeof(labels_path), "%s/labels.txt", data_dir);
    int n_labels = 0;
    char **label_names = NULL;
    {
        FILE *f = fopen(labels_path, "r");
        if (f) {
            char buf[256];
            int cap = 64;
            label_names = malloc(cap * sizeof(char*));
            while (fgets(buf, sizeof(buf), f)) {
                size_t len = strlen(buf);
                if (len > 0 && buf[len-1] == '\n') buf[len-1] = '\0';
                if (n_labels >= cap) { cap *= 2; label_names = realloc(label_names, cap * sizeof(char*)); }
                label_names[n_labels] = strdup(buf);
                n_labels++;
            }
            fclose(f);
        }
        printf("Labels: %d classes\n", n_labels);
    }

    /* ── Setup buckets ── */
    int bucket_limits[] = {33, 65, 97, 129};
    int bucket_starts[5];
    imagenet_vlm_dl_bucket(dl, 4, bucket_limits, bucket_starts);

    /* ══════════════════════════════════════════════════════════════
     *  TEST 1: Per-layer gradient norms (backward pass)
     * ══════════════════════════════════════════════════════════════ */

    printf("\n══════════ TEST 1: Per-layer gradient norms ══════════\n");

    int B = 4;
    int T128 = 128;
    dl->pos = bucket_starts[3];
    tensor *img_big = tensor_scratch(ctx.scratch, 4, (int[]){B, IMG_C, IMG_H, IMG_W}, 0);
    tensor *inp_big = tensor_zeros_data(ctx.data, 2, (int[]){B, T128});
    tensor *tgt_big = tensor_zeros_data(ctx.data, 2, (int[]){B, T128});
    tensor *msk_big = tensor_scratch(ctx.scratch, 2, (int[]){B, T128}, 0);
    int tl_big[B], lbl_big[B];
    int got = imagenet_vlm_dl_next_batch(dl, img_big, inp_big, tgt_big, msk_big,
                                          tl_big, lbl_big, B);
    if (got <= 0) { fprintf(stderr, "Test 1: batch error\n"); return 1; }
    printf("Batch got=%d  text_lens=[", got);
    for (int i = 0; i < got; i++) printf("%s%d", i?",":"", tl_big[i]);
    printf("]\n");
    if (label_names && lbl_big[0] >= 0 && lbl_big[0] < n_labels)
        printf("  Sample 0: %s\n", label_names[lbl_big[0]]);

    float gn;
    tensor *loss = vision_lm_train_step_padded(ctx.scratch, vlm,
                                                img_big, inp_big, tgt_big, msk_big,
                                                tl_big, NULL, GRAD_CLIP, &gn);
    printf("Loss=%.6f  grad_norm=%.4e\n", tensor_data_ptr(loss)[0], gn);
    print_grad_norms(&vlm->base, "Full VLM grad norms");
    mem_pool_reset(ctx.scratch);
    mem_pool_reset(ctx.data);

    /* ══════════════════════════════════════════════════════════════
     *  TEST 2: Image influence — compare real vs blank image
     * ══════════════════════════════════════════════════════════════ */

    printf("\n══════════ TEST 2: Image influence (real vs blank) ══════════\n");

    int T32 = 32;
    dl->pos = bucket_starts[0];
    tensor *img2 = tensor_scratch(ctx.scratch, 4, (int[]){1, IMG_C, IMG_H, IMG_W}, 0);
    tensor *inp2 = tensor_zeros_data(ctx.data, 2, (int[]){1, T32});
    tensor *tgt2 = tensor_zeros_data(ctx.data, 2, (int[]){1, T32});
    tensor *msk2 = tensor_scratch(ctx.scratch, 2, (int[]){1, T32}, 0);
    int tl2[1], lbl2[1];
    got = imagenet_vlm_dl_next_batch(dl, img2, inp2, tgt2, msk2, tl2, lbl2, 1);
    if (got <= 0) { fprintf(stderr, "Test 2: batch error\n"); return 1; }

    tensor *blank = tensor_zeros_data(ctx.data, 4, (int[]){1, IMG_C, IMG_H, IMG_W});
    int actual_T = tl2[0];
    tensor *inp_trim = tensor_slice(ctx.data, inp2, 1, 0, actual_T);
    inp_trim = tensor_contiguous(ctx.data, inp_trim);

    if (label_names && lbl2[0] >= 0 && lbl2[0] < n_labels)
        printf("Sample: label=%s  text_len=%d\n", label_names[lbl2[0]], actual_T);

    test_image_influence(ctx.scratch, ctx.data, vlm, img2, blank, inp_trim);
    mem_pool_reset(ctx.scratch);
    mem_pool_reset(ctx.data);

    /* ══════════════════════════════════════════════════════════════
     *  TEST 3: Bucket crossover — loss on T=32 after training on T=128
     * ══════════════════════════════════════════════════════════════ */

    printf("\n══════════ TEST 3: Bucket crossover loss ══════════\n");

    int B3 = 8;
    dl->pos = bucket_starts[3];
    tensor *img_b3 = tensor_scratch(ctx.scratch, 4, (int[]){B3, IMG_C, IMG_H, IMG_W}, 0);
    tensor *inp_b3 = tensor_zeros_data(ctx.data, 2, (int[]){B3, T128});
    tensor *tgt_b3 = tensor_zeros_data(ctx.data, 2, (int[]){B3, T128});
    tensor *msk_b3 = tensor_scratch(ctx.scratch, 2, (int[]){B3, T128}, 0);
    int tl_b3[B3], lbl_b3[B3];
    got = imagenet_vlm_dl_next_batch(dl, img_b3, inp_b3, tgt_b3, msk_b3, tl_b3, lbl_b3, B3);
    if (got > 0) {
        float l3 = forward_loss(ctx.scratch, ctx.data, vlm, img_b3, inp_b3,
                                 tgt_b3, msk_b3, tl_b3, got, T128);
        printf("Bucket 3 (T=128): loss=%.6f\n", l3);
    }

    dl->pos = bucket_starts[0];
    tensor *img_b0 = tensor_scratch(ctx.scratch, 4, (int[]){B3, IMG_C, IMG_H, IMG_W}, 0);
    tensor *inp_b0 = tensor_zeros_data(ctx.data, 2, (int[]){B3, T32});
    tensor *tgt_b0 = tensor_zeros_data(ctx.data, 2, (int[]){B3, T32});
    tensor *msk_b0 = tensor_scratch(ctx.scratch, 2, (int[]){B3, T32}, 0);
    int tl_b0[B3], lbl_b0[B3];
    got = imagenet_vlm_dl_next_batch(dl, img_b0, inp_b0, tgt_b0, msk_b0, tl_b0, lbl_b0, B3);
    if (got > 0) {
        float l0 = forward_loss(ctx.scratch, ctx.data, vlm, img_b0, inp_b0,
                                 tgt_b0, msk_b0, tl_b0, got, T32);
        printf("Bucket 0 (T=32):  loss=%.6f\n", l0);
    }

    /* Also test T=128 on bucket 0 data (longer sequence for short labels) */
    if (got > 0) {
        tensor *inp_b0_128 = tensor_zeros_data(ctx.data, 2, (int[]){B3, T128});
        tensor *tgt_b0_128 = tensor_zeros_data(ctx.data, 2, (int[]){B3, T128});
        tensor *msk_b0_128 = tensor_scratch(ctx.scratch, 2, (int[]){B3, T128}, 0);
        int tl_b0_128[B3];
        /* Copy data long-ways */
        for (int b = 0; b < got; b++) {
            int *in_src = (int*)tensor_data_ptr(inp_b0) + b * T32;
            int *t_src  = (int*)tensor_data_ptr(tgt_b0) + b * T32;
            float *m_src = (float*)tensor_data_ptr(msk_b0) + b * T32;
            int *in_dst = (int*)tensor_data_ptr(inp_b0_128) + b * T128;
            int *t_dst  = (int*)tensor_data_ptr(tgt_b0_128) + b * T128;
            float *m_dst = (float*)tensor_data_ptr(msk_b0_128) + b * T128;
            for (int t = 0; t < T32; t++) {
                in_dst[t] = in_src[t];
                t_dst[t]  = t_src[t];
                m_dst[t]  = m_src[t];
            }
            for (int t = T32; t < T128; t++) {
                in_dst[t] = IMAGENET_PAD_ID;
                t_dst[t]  = IMAGENET_PAD_ID;
                m_dst[t]  = 0.0f;
            }
            tl_b0_128[b] = tl_b0[b] > T128 ? T128 : tl_b0[b];  /* clamp */
        }
        float l0_128 = forward_loss(ctx.scratch, ctx.data, vlm, img_b0,
                                     inp_b0_128, tgt_b0_128, msk_b0_128,
                                     tl_b0_128, got, T128);
        printf("Bucket 0 data, T=128:  loss=%.6f  (same labels, longer padding)\n", l0_128);
    }

    mem_pool_reset(ctx.data);

    /* ══════════════════════════════════════════════════════════════
     *  TEST 4: Per-position logit analysis (entropy, argmax, P(EOS))
     * ══════════════════════════════════════════════════════════════ */

    printf("\n══════════ TEST 4: Per-position logit stats ══════════\n");

    {
        dnn_grad_ctx ng = dnn_no_grad_enter();
        dl->pos = bucket_starts[0];
        tensor *img4 = tensor_scratch(ctx.scratch, 4, (int[]){1, IMG_C, IMG_H, IMG_W}, 0);
        tensor *inp4 = tensor_zeros_data(ctx.data, 2, (int[]){1, T32});
        tensor *tgt4 = tensor_zeros_data(ctx.data, 2, (int[]){1, T32});
        tensor *msk4 = tensor_scratch(ctx.scratch, 2, (int[]){1, T32}, 0);
        int tl4[1], lbl4[1];
        got = imagenet_vlm_dl_next_batch(dl, img4, inp4, tgt4, msk4, tl4, lbl4, 1);
        if (got > 0) {
            int actual = tl4[0];
            int *inp_d = (int*)tensor_data_ptr(inp4);
            int *tgt_d = (int*)tensor_data_ptr(tgt4);
            /* float *msk_d = (float*)tensor_data_ptr(msk4); */

            /* Decode input for display */
            printf("  input[0..%d]: ", actual);
            for (int t = 0; t < actual; t++) {
                int id = inp_d[t];
                if (id < 256) printf("%c", id);
                else printf("[%s]", id==257?"BOS":id==258?"EOS":id==259?"PAD":"?");
            }
            printf("\n");
            printf("  target[0..%d]: ", actual);
            for (int t = 0; t < actual; t++) {
                int id = tgt_d[t];
                if (id < 256) printf("%c", id);
                else printf("[%s]", id==258?"EOS":id==259?"PAD":"?");
            }
            printf("\n");

            tensor *logits = vision_lm_forward(ctx.scratch, vlm, img4, inp4);
            float *ld = tensor_data_ptr(logits);
            int total_seq = logits->shape[1];
            int V = logits->shape[2];
            printf("  t(pos)  entropy  argmax  P(EOS)    expected  match?\n");
            for (int t = I; t < total_seq; t++) {
                float *row = ld + t * V;
                float mx = row_max_val(row, V);
                float se = row_softmax_denom(row, V, mx);
                float inv_se = 1.0f / se;
                float H = 0.0f;
                int argmax = 0;
                float p_eos = 0.0f;
                for (int v = 0; v < V; v++) {
                    float p = expf(row[v] - mx) * inv_se;
                    if (p > 1e-10f) H -= p * log2f(p);
                    if (expf(row[v] - mx) * inv_se > expf(row[argmax] - mx) * inv_se) argmax = v;
                    if (v == TOKENIZER_EOS_ID) p_eos = p;
                }
                int pos = t - I;
                int expected = (pos < actual) ? tgt_d[pos] : -1;
                int correct = (expected >= 0 && argmax == expected) ? 1 : 0;
                printf("  I+%2d/%2d  %.3f  %-4s  %.2e  %-4s  %s\n",
                       pos, actual, H,
                       argmax<256?(char[]){argmax,0}:(argmax==257?"BOS":argmax==258?"EOS":"--"),
                       p_eos,
                       expected==258?"EOS":expected<256?(char[]){expected,0}:"--",
                        expected < 0 ? "n/a" : (correct ? "OK" : "WRONG"));
            }
        }
        mem_pool_reset(ctx.scratch);
        mem_pool_reset(ctx.data);
        dnn_no_grad_exit(ng);
    }

    /* ══════════════════════════════════════════════════════════════
     *  TEST 5: Autoregressive generation (one sample, step-by-step)
     * ══════════════════════════════════════════════════════════════ */

    printf("\n══════════ TEST 5: Step-by-step generation ══════════\n");

    {
        dnn_grad_ctx ng = dnn_no_grad_enter();
        dl->pos = bucket_starts[0];
        tensor *img5 = tensor_scratch(ctx.scratch, 4, (int[]){1, IMG_C, IMG_H, IMG_W}, 0);
        tensor *inp5 = tensor_zeros_data(ctx.data, 2, (int[]){1, T32});
        tensor *tgt5 = tensor_zeros_data(ctx.data, 2, (int[]){1, T32});
        tensor *msk5 = tensor_scratch(ctx.scratch, 2, (int[]){1, T32}, 0);
        int tl5[1], lbl5[1];
        got = imagenet_vlm_dl_next_batch(dl, img5, inp5, tgt5, msk5, tl5, lbl5, 1);
        if (got > 0 && label_names && lbl5[0] >= 0 && lbl5[0] < n_labels) {
            const char *truth = label_names[lbl5[0]];
            printf("  Truth: \"%s\" (label %d, text_len=%d)\n", truth, lbl5[0], tl5[0]);

            int max_new = 64;
            int tokens[1 + max_new + 1]; /* BOS + max_new */
            tokens[0] = TOKENIZER_BOS_ID;
            int cur_len = 1;

            while (cur_len < max_new) {
                tensor *prompt = tensor_zeros_data(ctx.data, 2, (int[]){1, cur_len});
                int *pd = (int*)tensor_data_ptr(prompt);
                for (int t = 0; t < cur_len; t++) pd[t] = tokens[t];

                tensor *logits = vision_lm_forward(ctx.scratch, vlm, img5, prompt);
                float *ld = tensor_data_ptr(logits);
                int last_pos = I + cur_len - 1;
                float *last_row = ld + last_pos * vlm->vocab_size;

                /* Compute softmax stats */
                float mx = row_max_val(last_row, vlm->vocab_size);
                float se = row_softmax_denom(last_row, vlm->vocab_size, mx);
                float inv_se = 1.0f / se;

                int pred = 0;
                for (int v = 1; v < vlm->vocab_size; v++)
                    if (last_row[v] > last_row[pred]) pred = v;

                float p_pred = expf(last_row[pred] - mx) * inv_se;
                float p_eos  = expf(last_row[TOKENIZER_EOS_ID] - mx) * inv_se;
                float H = 0.0f;
                for (int v = 0; v < vlm->vocab_size; v++) {
                    float p = expf(last_row[v] - mx) * inv_se;
                    if (p > 1e-10f) H -= p * log2f(p);
                }

                /* Expected token at this position (if any) */
                int expected = -1;
                if (cur_len - 1 < tl5[0]) {
                    int *tgt_d = (int*)tensor_data_ptr(tgt5);
                    expected = tgt_d[cur_len - 1];
                }

                char pred_str[8], exp_str[8];
                if (pred < 256) { pred_str[0] = pred; pred_str[1] = '\0'; }
                else if (pred == 257) strcpy(pred_str, "BOS");
                else if (pred == 258) strcpy(pred_str, "EOS");
                else snprintf(pred_str, sizeof(pred_str), "%d", pred);
                if (expected < 0) strcpy(exp_str, "-");
                else if (expected < 256) { exp_str[0] = expected; exp_str[1] = '\0'; }
                else if (expected == 258) strcpy(exp_str, "EOS");
                else snprintf(exp_str, sizeof(exp_str), "%d", expected);

                printf("  step%3d  H=%.3f  argmax=%-4s(p=%.3f)  P(EOS)=%.2e  expected=%-4s  %s\n",
                       cur_len - 1, H, pred_str, (double)p_pred, p_eos,
                       exp_str, expected >= 0 && pred == expected ? "OK" : "");

                tokens[cur_len] = pred;
                cur_len++;
                if (pred == TOKENIZER_EOS_ID) break;
                mem_pool_reset(ctx.scratch);
                mem_pool_reset(ctx.data);
            }

            /* Decode generated string */
            char decoded[128];
            int dn = 0;
            for (int t = 1; t < cur_len; t++) {
                int tok = tokens[t];
                if (tok == TOKENIZER_EOS_ID) break;
                if (tok < 0 || tok > 255) break;
                if (dn < (int)sizeof(decoded) - 1)
                    decoded[dn++] = (tok >= 32) ? (char)tok : '?';
            }
            decoded[dn] = '\0';
            printf("  Generated: \"%s\"\n", decoded);
            printf("  Match: %s\n", strcmp(decoded, truth) == 0 ? "YES" : "NO");
        }
        mem_pool_reset(ctx.scratch);
        mem_pool_reset(ctx.data);
        dnn_no_grad_exit(ng);
    }

    /* ══════════════════════════════════════════════════════════════
     *  TEST 6: Gradient flow to image encoder
     * ══════════════════════════════════════════════════════════════ */

    printf("\n══════════ TEST 6: Gradient flow to image encoder ══════════\n");

    {
        dl->pos = bucket_starts[0];
        tensor *img6 = tensor_scratch(ctx.scratch, 4, (int[]){1, IMG_C, IMG_H, IMG_W}, 0);
        tensor *inp6 = tensor_zeros_data(ctx.data, 2, (int[]){1, T32});
        tensor *tgt6 = tensor_zeros_data(ctx.data, 2, (int[]){1, T32});
        tensor *msk6 = tensor_scratch(ctx.scratch, 2, (int[]){1, T32}, 0);
        int tl6[1], lbl6[1];
        got = imagenet_vlm_dl_next_batch(dl, img6, inp6, tgt6, msk6, tl6, lbl6, 1);
        if (got > 0) {
            /* Manual forward + backward with clean gradients */
            tensor *embeds = vision_lm_build_embeds(ctx.scratch, vlm, img6, inp6);
            tensor *logits = decoder_lm_forward_embeds_ex(ctx.scratch, vlm->lm, embeds,
                                                          ATTENTION_PREFIX_LM, I, NULL);
            tensor *lt = tensor_slice(ctx.scratch, logits, 1, I, T32);
            tensor *loss6 = tensor_cross_entropy_masked(ctx.scratch, lt, tgt6, msk6, 2);

            /* Zero grads manually */
            int np;
            tensor **params = module_parameters(&vlm->base, &np);
            for (int i = 0; i < np; i++) {
                float *g = tensor_grad(params[i]);
                if (g) memset(g, 0, (size_t)tensor_numel(params[i]) * sizeof(float));
            }

            dnn_backward(ctx.scratch, loss6);

            printf("  Loss=%.6f\n", tensor_data_ptr(loss6)[0]);

            /* Check critical gradients */
            float *pw_grad = tensor_grad(vlm->patch_embed->weight);
            float *ew_grad = tensor_grad(vlm->lm->embed->weight);
            double pn = 0.0, en = 0.0;
            int pnc = tensor_numel(vlm->patch_embed->weight);
            int enc = tensor_numel(vlm->lm->embed->weight);
            for (int i = 0; i < pnc; i++) pn += (double)pw_grad[i] * (double)pw_grad[i];
            for (int i = 0; i < enc; i++) en += (double)ew_grad[i] * (double)ew_grad[i];

            int pz = 0;
            for (int i = 0; i < pnc; i++) if (pw_grad[i] == 0.0f) pz++;
            printf("  patch_embed.weight gn=%.4e  (%d/%d zero=%.1f%%)\n",
                   sqrtf((float)pn), pz, pnc, 100.0 * pz / (pnc > 0 ? pnc : 1));

            int ez = 0;
            for (int i = 0; i < enc; i++) if (ew_grad[i] == 0.0f) ez++;
            printf("  embed.weight       gn=%.4e  (%d/%d zero=%.1f%%)\n",
                   sqrtf((float)en), ez, enc, 100.0 * ez / (enc > 0 ? enc : 1));

            printf("  ratio patch/embed  gn=%.4e\n",
                   sqrtf((float)pn) / (sqrtf((float)en) + 1e-10f));

            /* Check image_pos gradients */
            if (vlm->image_pos) {
                float *ip_grad = tensor_grad(vlm->image_pos);
                double ipn = 0.0;
                int ipc = tensor_numel(vlm->image_pos);
                for (int i = 0; i < ipc; i++) ipn += (double)ip_grad[i] * (double)ip_grad[i];
                int ipz = 0;
                for (int i = 0; i < ipc; i++) if (ip_grad[i] == 0.0f) ipz++;
                printf("  image_pos          gn=%.4e  (%d/%d zero=%.1f%%)\n",
                       sqrtf((float)ipn), ipz, ipc, 100.0 * ipz / (ipc > 0 ? ipc : 1));
            }

            /* Patch embed bias */
            float *pb_grad = tensor_grad(vlm->patch_embed->bias);
            double pbn = 0.0;
            int pbc = tensor_numel(vlm->patch_embed->bias);
            for (int i = 0; i < pbc; i++) pbn += (double)pb_grad[i] * (double)pb_grad[i];
            printf("  patch_embed.bias   gn=%.4e\n", sqrtf((float)pbn));

            /* Image norm */
            if (vlm->image_norm->weight) {
                float *inw_grad = tensor_grad(vlm->image_norm->weight);
                double inn = 0.0;
                int inc = tensor_numel(vlm->image_norm->weight);
                for (int i = 0; i < inc; i++) inn += (double)inw_grad[i] * (double)inw_grad[i];
                printf("  image_norm.weight  gn=%.4e\n", sqrtf((float)inn));
            }

            /* LM sub-module breakdown */
            print_grad_norms(&vlm->lm->base, "LM grad norms");
        }
        mem_pool_reset(ctx.scratch);
        mem_pool_reset(ctx.data);
    }

    /* Cleanup */
    imagenet_vlm_dl_free(dl);
    if (label_names) {
        for (int i = 0; i < n_labels; i++) free(label_names[i]);
        free(label_names);
    }
    dnn_ctx_destroy(&ctx);
    printf("\n=== vlm_debug_2 complete ===\n");
    return 0;
}
