/* ══════════════════════════════════════════════════════════════════
 * vlm_probe — checkpoint-grounded ImageNet VLM diagnostics
 *
 * Measures candidate-label scoring accuracy/ranks and real-vs-blank image
 * dependence. Intended for post-hoc debug of examples/imagenet_vlm.
 * ══════════════════════════════════════════════════════════════════ */

#include "dnn.h"
#include "gpt.h"
#include "vlm.h"
#include "imagenet_vlm.h"
#include "tokenizer.h"
#include <float.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/param.h>

#define IMG_H 224
#define IMG_W 224
#define IMG_C 3
#define PATCH_SIZE 16
#define D_MODEL 128
#define N_LAYERS 2
#define N_HEADS 2
#define D_K 64
#define INTERMEDIATE 256
#define VOCAB_SIZE 261
#define TMAX 128
#define CHUNK 24

static char **load_labels(const char *path, int *n_out) {
    FILE *f = fopen(path, "r");
    if (!f) return NULL;
    int cap = 1024, n = 0;
    char **names = (char **)malloc((size_t)cap * sizeof(char *));
    char buf[512];
    while (fgets(buf, sizeof(buf), f)) {
        size_t len = strlen(buf);
        while (len && (buf[len - 1] == '\n' || buf[len - 1] == '\r')) buf[--len] = '\0';
        if (n >= cap) {
            cap *= 2;
            names = (char **)realloc(names, (size_t)cap * sizeof(char *));
        }
        names[n++] = strdup(buf);
    }
    fclose(f);
    *n_out = n;
    return names;
}

static void fill_label_batch(tensor *inp, tensor *tgt, tensor *msk,
                             int *text_lens, char **labels,
                             int start, int count) {
    int *id = (int *)tensor_data_ptr(inp);
    int *td = (int *)tensor_data_ptr(tgt);
    float *md = (float *)tensor_data_ptr(msk);
    int T = inp->shape[1];
    for (int b = 0; b < count; b++) {
        const unsigned char *s = (const unsigned char *)labels[start + b];
        int len = (int)strlen((const char *)s);
        if (len + 1 > T) len = T - 1;
        int *ir = id + (long)b * T;
        int *tr = td + (long)b * T;
        float *mr = md + (long)b * T;
        for (int t = 0; t < T; t++) {
            ir[t] = TOKENIZER_PAD_ID;
            tr[t] = TOKENIZER_PAD_ID;
            mr[t] = 0.0f;
        }
        ir[0] = TOKENIZER_BOS_ID;
        for (int t = 0; t < len; t++) {
            ir[t + 1] = (int)s[t];
            tr[t] = (int)s[t];
            mr[t] = 1.0f;
        }
        tr[len] = TOKENIZER_EOS_ID;
        mr[len] = 1.0f;
        text_lens[b] = len + 2; /* BOS + bytes + EOS */
    }
}

static float ce_row(const float *row, int V, int target) {
    float mx = -INFINITY;
    for (int v = 0; v < V; v++) if (row[v] > mx) mx = row[v];
    float se = 0.0f;
    for (int v = 0; v < V; v++) se += expf(row[v] - mx);
    return logf(se) + mx - row[target];
}

static void score_chunk(struct mem_pool *scratch, vision_lm *vlm,
                        const float *raw_img, int blank,
                        tensor *inp, tensor *tgt, tensor *msk,
                        int *text_lens, int count, float *out_loss) {
    tensor *img = tensor_scratch(scratch, 4, (int[]){count, IMG_C, IMG_H, IMG_W}, 0);
    float *im = (float *)tensor_data_ptr(img);
    long npx = (long)IMG_C * IMG_H * IMG_W;
    for (int b = 0; b < count; b++) {
        if (blank) memset(im + b * npx, 0, (size_t)npx * sizeof(float));
        else memcpy(im + b * npx, raw_img, (size_t)npx * sizeof(float));
    }

    int I = vlm->n_img_tokens;
    int *combined_lens = (int *)_mem_pool_alloc(scratch, (size_t)count * sizeof(int), NULL);
    for (int b = 0; b < count; b++) combined_lens[b] = I + text_lens[b];

    tensor *embeds = vision_lm_build_embeds(scratch, vlm, img, inp);
    tensor *logits = decoder_lm_forward_embeds_ex(scratch, vlm->lm, embeds,
                                                  ATTENTION_PREFIX_LM, I, combined_lens);
    float *ld = (float *)tensor_data_ptr(logits);
    int V = vlm->vocab_size;
    int T = inp->shape[1];
    int *td = (int *)tensor_data_ptr(tgt);
    float *md = (float *)tensor_data_ptr(msk);
    for (int b = 0; b < count; b++) {
        float sum = 0.0f;
        int ntok = 0;
        for (int t = 0; t < T; t++) {
            if (md[(long)b * T + t] == 0.0f) continue;
            const float *row = ld + (long)b * logits->strides[0] + (long)(I + t) * V;
            sum += ce_row(row, V, td[(long)b * T + t]);
            ntok++;
        }
        out_loss[b] = ntok ? sum / (float)ntok : 1e30f;
    }
}

static int rank_of_true(const float *losses, int n, int true_id) {
    int rank = 1;
    float tl = losses[true_id];
    for (int i = 0; i < n; i++) if (losses[i] < tl) rank++;
    return rank;
}

static int argmin(const float *x, int n) {
    int best = 0;
    for (int i = 1; i < n; i++) if (x[i] < x[best]) best = i;
    return best;
}

int main(int argc, char **argv) {
    const char *data_dir = NULL;
    const char *ckpt = NULL;
    int max_samples = 12;
    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "--data-dir") == 0 && i + 1 < argc) data_dir = argv[++i];
        else if (strcmp(argv[i], "--ckpt") == 0 && i + 1 < argc) ckpt = argv[++i];
        else if (strcmp(argv[i], "--n") == 0 && i + 1 < argc) max_samples = atoi(argv[++i]);
    }
    if (!data_dir || !ckpt) {
        fprintf(stderr, "Usage: %s --data-dir PATH --ckpt PATH [--n N]\n", argv[0]);
        return 1;
    }

    dnn_ctx ctx;
    dnn_ctx_init(&ctx, 256 * 1024 * 1024, (size_t)4096 * 1024 * 1024, 256 * 1024 * 1024);
    srand(42);
    vision_lm *vlm = vision_lm_create(ctx.params, VOCAB_SIZE, D_MODEL,
                                       N_LAYERS, N_HEADS, D_K, INTERMEDIATE,
                                       IMG_C, IMG_H, IMG_W, PATCH_SIZE, 1);
    vision_lm_init_weights(vlm);
    vision_lm_enable_rope(ctx.params, vlm, TMAX, 10000.0f);
    module_load(&vlm->base, ckpt, 1);

    char labels_path[MAXPATHLEN];
    snprintf(labels_path, sizeof(labels_path), "%s/labels.txt", data_dir);
    int n_labels = 0;
    char **labels = load_labels(labels_path, &n_labels);
    if (!labels || n_labels <= 0) { fprintf(stderr, "labels load failed\n"); return 1; }

    imagenet_vlm_dl *val = imagenet_vlm_dl_create("val", data_dir, 0, 999);
    if (!val) { fprintf(stderr, "val dl failed\n"); return 1; }

    printf("ckpt=%s\n", ckpt);
    printf("val_samples=%d labels=%d probe_n=%d chunk=%d T=%d\n", val->total_samples, n_labels, max_samples, CHUNK, TMAX);

    int correct_real = 0, correct_blank = 0, top5_real = 0, top5_blank = 0;
    double rank_sum_real = 0.0, rank_sum_blank = 0.0;
    double true_loss_real_sum = 0.0, true_loss_blank_sum = 0.0;

    dnn_grad_ctx ng = dnn_no_grad_enter();
    int n_done = 0;
    for (; n_done < max_samples; n_done++) {
        tensor *img1 = tensor_scratch(ctx.scratch, 4, (int[]){1, IMG_C, IMG_H, IMG_W}, 0);
        tensor *inp1 = tensor_zeros_data(ctx.data, 2, (int[]){1, TMAX});
        tensor *tgt1 = tensor_zeros_data(ctx.data, 2, (int[]){1, TMAX});
        tensor *msk1 = tensor_scratch(ctx.scratch, 2, (int[]){1, TMAX}, 0);
        int tl1[1], lbl1[1];
        int got = imagenet_vlm_dl_next_batch(val, img1, inp1, tgt1, msk1, tl1, lbl1, 1);
        if (got <= 0) break;

        float raw_img[IMG_C * IMG_H * IMG_W];
        memcpy(raw_img, tensor_data_ptr(img1), sizeof(raw_img));
        int true_id = lbl1[0];

        float *loss_real = (float *)malloc((size_t)n_labels * sizeof(float));
        float *loss_blank = (float *)malloc((size_t)n_labels * sizeof(float));
        for (int start = 0; start < n_labels; start += CHUNK) {
            int count = n_labels - start < CHUNK ? n_labels - start : CHUNK;
            tensor *inp = tensor_zeros_data(ctx.data, 2, (int[]){count, TMAX});
            tensor *tgt = tensor_zeros_data(ctx.data, 2, (int[]){count, TMAX});
            tensor *msk = tensor_scratch(ctx.scratch, 2, (int[]){count, TMAX}, 0);
            int *tls = (int *)_mem_pool_alloc(ctx.scratch, (size_t)count * sizeof(int), NULL);
            fill_label_batch(inp, tgt, msk, tls, labels, start, count);
            score_chunk(ctx.scratch, vlm, raw_img, 0, inp, tgt, msk, tls, count, loss_real + start);
            mem_pool_reset(ctx.scratch);

            inp = tensor_zeros_data(ctx.data, 2, (int[]){count, TMAX});
            tgt = tensor_zeros_data(ctx.data, 2, (int[]){count, TMAX});
            msk = tensor_scratch(ctx.scratch, 2, (int[]){count, TMAX}, 0);
            tls = (int *)_mem_pool_alloc(ctx.scratch, (size_t)count * sizeof(int), NULL);
            fill_label_batch(inp, tgt, msk, tls, labels, start, count);
            score_chunk(ctx.scratch, vlm, raw_img, 1, inp, tgt, msk, tls, count, loss_blank + start);
            mem_pool_reset(ctx.scratch);
            mem_pool_reset(ctx.data);
        }

        int pred_r = argmin(loss_real, n_labels);
        int pred_b = argmin(loss_blank, n_labels);
        int rank_r = rank_of_true(loss_real, n_labels, true_id);
        int rank_b = rank_of_true(loss_blank, n_labels, true_id);
        correct_real += (pred_r == true_id);
        correct_blank += (pred_b == true_id);
        top5_real += (rank_r <= 5);
        top5_blank += (rank_b <= 5);
        rank_sum_real += rank_r;
        rank_sum_blank += rank_b;
        true_loss_real_sum += loss_real[true_id];
        true_loss_blank_sum += loss_blank[true_id];

        printf("sample=%02d true=%d %-28s real_pred=%d rank=%d true_loss=%.4f pred_loss=%.4f blank_pred=%d blank_rank=%d blank_true_loss=%.4f\n",
               n_done, true_id, labels[true_id], pred_r, rank_r, loss_real[true_id], loss_real[pred_r],
               pred_b, rank_b, loss_blank[true_id]);

        free(loss_real);
        free(loss_blank);
        mem_pool_reset(ctx.scratch);
        mem_pool_reset(ctx.data);
    }
    dnn_no_grad_exit(ng);

    if (n_done > 0) {
        printf("SUMMARY n=%d real_top1=%.4f real_top5=%.4f blank_top1=%.4f blank_top5=%.4f mean_rank_real=%.2f mean_rank_blank=%.2f mean_true_loss_real=%.4f mean_true_loss_blank=%.4f\n",
               n_done,
               (float)correct_real / n_done,
               (float)top5_real / n_done,
               (float)correct_blank / n_done,
               (float)top5_blank / n_done,
               rank_sum_real / n_done,
               rank_sum_blank / n_done,
               true_loss_real_sum / n_done,
               true_loss_blank_sum / n_done);
    }

    imagenet_vlm_dl_free(val);
    for (int i = 0; i < n_labels; i++) free(labels[i]);
    free(labels);
    dnn_ctx_destroy(&ctx);
    return 0;
}
