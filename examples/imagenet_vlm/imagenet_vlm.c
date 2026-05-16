/* ══════════════════════════════════════════════════════════════════
 *  imagenet_vlm — train VLM on ImageNet-1k
 *
 *  Treats classification as label-text generation with byte-level
 *  tokenizer, prefix-LM attention, bucketed padded batches.
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

/* ── Config ── */
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
#define BATCH_SIZE       64
#define MAX_EPOCHS        10
#define LR              5e-4f
#define MIN_LR          5e-5f
#define GRAD_CLIP        5.0f
#define WARMUP_EPOCHS     2

#define N_BUCKETS         4
/* bucket_limits are exclusive upper bounds on text_lens (stored + BOS).
 * T per bucket must be >= bucket_limit - 1.
 * Longest synset name: 121 bytes + EOS = 122 stored; +BOS = 123.
 * Bucket 3 T=128 covers all. */
#define BUCKET_LIMITS    {33, 65, 97, 129}
#define BUCKET_T         {32, 64, 96, 128}

static const int BUCKET_TMAX[N_BUCKETS] = BUCKET_T;

/* ══════════════════════════════════════════════════════════════════
 *  eval_full_string — validation via full autoregressive label decode
 * ══════════════════════════════════════════════════════════════════ */

static float eval_full_string(struct mem_pool *scratch, struct mem_pool *data,
                               vision_lm *vlm, imagenet_vlm_dl *dl,
                               const char *names[], int n_names, int max_n) {
    imagenet_vlm_dl_reset(dl);

    int correct = 0, total = 0;
    dnn_grad_ctx ng = dnn_no_grad_enter();

    const int BS = 10;
    while (total < max_n) {
        int take = max_n - total < BS ? max_n - total : BS;
        tensor *img = tensor_zeros_data(data, 4,
                                        (int[]){take, IMG_C, IMG_H, IMG_W});
        tensor *inp = tensor_zeros_data(data, 2, (int[]){take, IMAGENET_MAX_TEXT_LEN});
        tensor *tgt = tensor_zeros_data(data, 2, (int[]){take, IMAGENET_MAX_TEXT_LEN});
        tensor *msk = tensor_scratch(scratch, 2, (int[]){take, IMAGENET_MAX_TEXT_LEN}, 0);
        int tl[take], lbl[take];

        int got = imagenet_vlm_dl_next_batch(dl, img, inp, tgt, msk, tl, lbl, take);
        if (got < 0) {
            mem_pool_reset(scratch);
            mem_pool_reset(data);
            dnn_no_grad_exit(ng);
            return -1.0f;
        }
        if (got == 0) {
            mem_pool_reset(scratch);
            mem_pool_reset(data);
            break;
        }

        int max_decode = IMAGENET_MAX_TEXT_LEN - 1;
        int tokens[take][IMAGENET_MAX_TEXT_LEN];
        int finished[take];
        for (int i = 0; i < take; i++) {
            for (int t = 0; t < IMAGENET_MAX_TEXT_LEN; t++)
                tokens[i][t] = TOKENIZER_PAD_ID;
            tokens[i][0] = (i < got) ? TOKENIZER_BOS_ID : TOKENIZER_PAD_ID;
            finished[i] = (i >= got);
        }

        int seq_len = 1;
        int all_done = 0;
        while (seq_len <= max_decode && !all_done) {
            tensor *prompt = tensor_zeros_data(data, 2, (int[]){take, seq_len});
            int *pd = (int *)tensor_data_ptr(prompt);
            for (int i = 0; i < take; i++)
                for (int t = 0; t < seq_len; t++)
                    pd[(long)i * seq_len + t] = tokens[i][t];

            tensor *logits = vision_lm_forward(scratch, vlm, img, prompt);
            float *ld = tensor_data_ptr(logits);

            all_done = 1;
            for (int i = 0; i < got; i++) {
                if (finished[i]) continue;
                int last_tok = N_IMG_TOK + seq_len - 1;
                float *row = ld + (long)i * logits->strides[0]
                              + (long)last_tok * vlm->vocab_size;
                int pred = 0;
                for (int v = 1; v < vlm->vocab_size; v++)
                    if (row[v] > row[pred]) pred = v;
                tokens[i][seq_len] = pred;
                if (pred == TOKENIZER_EOS_ID || seq_len == max_decode)
                    finished[i] = 1;
                else
                    all_done = 0;
            }
            seq_len++;
            mem_pool_reset(scratch);
        }

        for (int i = 0; i < got; i++) {
            char pred_name[IMAGENET_MAX_TEXT_LEN];
            int pn = 0;
            for (int t = 1; t < seq_len && t <= max_decode; t++) {
                int tok = tokens[i][t];
                if (tok == TOKENIZER_EOS_ID) break;
                if (tok < 0 || tok > 255) break;
                if (pn < (int)sizeof(pred_name) - 1)
                    pred_name[pn++] = (tok >= 32 && tok < 127) ? (char)tok : '?';
            }
            pred_name[pn] = '\0';
            if (lbl[i] >= 0 && lbl[i] < n_names && strcmp(pred_name, names[lbl[i]]) == 0)
                correct++;
            total++;
        }

        mem_pool_reset(scratch);
        mem_pool_reset(data);
    }

    imagenet_vlm_dl_reset(dl);
    dnn_no_grad_exit(ng);
    return total > 0 ? (float)correct / (float)total : 0.0f;
}

/* ══════════════════════════════════════════════════════════════════
 *  Load labels.txt into char** for prediction printing
 * ══════════════════════════════════════════════════════════════════ */

static char **load_labels(const char *path, int *n) {
    FILE *f = fopen(path, "r");
    if (!f) return NULL;
    char buf[256];
    int cap = 64, cnt = 0;
    char **names = malloc((size_t)cap * sizeof(char *));
    while (fgets(buf, sizeof(buf), f)) {
        size_t len = strlen(buf);
        if (len > 0 && buf[len - 1] == '\n') buf[len - 1] = '\0';
        if (cnt >= cap) {
            cap *= 2;
            names = realloc(names, (size_t)cap * sizeof(char *));
        }
        names[cnt] = strdup(buf);
        cnt++;
    }
    fclose(f);
    *n = cnt;
    return names;
}

/* ══════════════════════════════════════════════════════════════════
 *  show_preds — peek 10 samples, forward, print label vs prediction
 * ══════════════════════════════════════════════════════════════════ */

static void show_preds(struct mem_pool *scratch, struct mem_pool *data,
                        vision_lm *vlm, imagenet_vlm_dl *dl,
                        const char *names[]) {
    long saved_pos = dl->pos;
    int take = dl->total_samples < 10 ? dl->total_samples : 10;
    if (take <= 0) return;

    /* Random validation samples: temporarily patch shuffle_order[0:take]. */
    int saved_order[take];
    for (int i = 0; i < take; i++) {
        saved_order[i] = dl->shuffle_order[i];
        dl->shuffle_order[i] = rand() % dl->total_samples;
    }
    dl->pos = 0;

    int T_pred = IMAGENET_MAX_TEXT_LEN;
    tensor *img = tensor_zeros_data(data, 4,
                                  (int[]){take, IMG_C, IMG_H, IMG_W});
    tensor *inp = tensor_zeros_data(data, 2, (int[]){take, T_pred});
    tensor *tgt = tensor_zeros_data(data, 2, (int[]){take, T_pred});
    tensor *msk = tensor_scratch(scratch, 2, (int[]){take, T_pred}, 0);
    int tl[take], lbl[take];

    int got = imagenet_vlm_dl_next_batch(dl, img, inp, tgt, msk, tl, lbl, take);
    for (int i = 0; i < take; i++) dl->shuffle_order[i] = saved_order[i];
    dl->pos = saved_pos;
    if (got <= 0) {
        mem_pool_reset(scratch);
        mem_pool_reset(data);
        return;
    }

    dnn_grad_ctx ng = dnn_no_grad_enter();

    /* Autoregressive decode: predict one byte at a time until EOS. */
    int max_decode = IMAGENET_MAX_TEXT_LEN - 1;
    int tokens[take][IMAGENET_MAX_TEXT_LEN];  /* [i][0]=BOS, [i][1..]=predicted bytes */
    int finished[take];
    for (int i = 0; i < take; i++) {
        for (int t = 0; t < IMAGENET_MAX_TEXT_LEN; t++)
            tokens[i][t] = TOKENIZER_PAD_ID;
        tokens[i][0] = (i < got) ? TOKENIZER_BOS_ID : TOKENIZER_PAD_ID;
        finished[i] = (i >= got);
    }

    int seq_len = 1;
    int all_done = 0;
    while (seq_len <= max_decode && !all_done) {
        tensor *prompt = tensor_zeros_data(data, 2, (int[]){take, seq_len});
        int *pd = (int *)tensor_data_ptr(prompt);
        for (int i = 0; i < take; i++)
            for (int t = 0; t < seq_len; t++)
                pd[(long)i * seq_len + t] = tokens[i][t];

        tensor *logits = vision_lm_forward(scratch, vlm, img, prompt);
        float *ld = tensor_data_ptr(logits);

        all_done = 1;
        for (int i = 0; i < got; i++) {
            if (finished[i]) continue;
            int last_tok = N_IMG_TOK + seq_len - 1;
            float *row = ld + (long)i * logits->strides[0]
                          + (long)last_tok * vlm->vocab_size;
            int pred = 0;
            for (int v = 1; v < vlm->vocab_size; v++)
                if (row[v] > row[pred]) pred = v;
            tokens[i][seq_len] = pred;
            if (pred == TOKENIZER_EOS_ID || seq_len == max_decode)
                finished[i] = 1;
            else
                all_done = 0;
        }
        seq_len++;
        mem_pool_reset(scratch);
    }

    /* Decode predicted byte sequences to strings and print. */
    printf("  ── preds ──\n");
    for (int i = 0; i < got; i++) {
        char pred_name[IMAGENET_MAX_TEXT_LEN];
        int pn = 0;
        for (int t = 1; t < seq_len && t <= max_decode; t++) {
            int tok = tokens[i][t];
            if (tok == TOKENIZER_EOS_ID) break;
            if (tok < 0 || tok > 255) break;
            if (pn < (int)sizeof(pred_name) - 1)
                pred_name[pn++] = (tok >= 32 && tok < 127) ? (char)tok : '?';
        }
        pred_name[pn] = '\0';
        int ok = strcmp(pred_name, names ? names[lbl[i]] : "") == 0;
        printf("  [%d] true=%-20s VS pred=%-20s %s\n",
               lbl[i],
               names ? names[lbl[i]] : "?",
               pred_name,
               ok ? "OK" : "FAIL");
    }
    printf("  ──\n");

    mem_pool_reset(scratch);
    mem_pool_reset(data);
    dnn_no_grad_exit(ng);
}

/* ══════════════════════════════════════════════════════════════════
 *  Main
 * ══════════════════════════════════════════════════════════════════ */

int main(int argc, char **argv) {
    printf("=== ImageNet VLM (prefix-LM text-generation classification) ===\n\n");

    /* ── Parse args ── */
    const char *data_dir = NULL;
    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "--data-dir") == 0 && i + 1 < argc) {
            data_dir = argv[++i];
        } else if (strcmp(argv[i], "--help") == 0 || strcmp(argv[i], "-h") == 0) {
            printf("Usage: %s --data-dir PATH\n", argv[0]);
            return 0;
        }
    }
    if (!data_dir) {
        fprintf(stderr, "Error: --data-dir PATH required\n");
        return 1;
    }

    dnn_ctx ctx;
    dnn_ctx_init(&ctx, 256*1024*1024, 1024*1024*1024, 64*1024*1024);

    /* ── Dataloaders ── */
    imagenet_vlm_dl *train_dl = imagenet_vlm_dl_create("train", data_dir, 1, 42);
    if (!train_dl) { fprintf(stderr, "Failed to create train dataloader\n"); return 1; }

    imagenet_vlm_dl *val_dl = imagenet_vlm_dl_create("val", data_dir, 0, 999);
    /* val dataloader is optional; warn if missing but continue */

    /* ── Load label names ── */
    int n_labels = 0;
    char labels_path[MAXPATHLEN];
    snprintf(labels_path, sizeof(labels_path), "%s/labels.txt", data_dir);
    char **label_names = load_labels(labels_path, &n_labels);
    if (!label_names)
        printf("Labels: (none — run prep_imagenet_vlm.py first)\n");
    else
        printf("Labels: %d classes from %s\n", n_labels, labels_path);

    printf("Train: %d samples  Shards: %d\n",
           imagenet_vlm_dl_total(train_dl), train_dl->num_shards);
    if (val_dl)
        printf("Val:   %d samples  Shards: %d\n",
               imagenet_vlm_dl_total(val_dl), val_dl->num_shards);
    else
        printf("Val:   (none)\n");

    /* ── Create VLM ── */
    srand(42);
    vision_lm *vlm = vision_lm_create(ctx.params, VOCAB_SIZE, D_MODEL,
                                       N_LAYERS, N_HEADS, D_K, INTERMEDIATE,
                                       IMG_C, IMG_H, IMG_W, PATCH_SIZE, 1);
    vision_lm_init_weights(vlm);
    vision_lm_enable_rope(ctx.params, vlm, 128, 10000.0f);

    printf("VLM: %.2fM params  patches=%d  layers=%d  D=%d\n",
           vision_lm_num_parameters(vlm) / 1e6, N_IMG_TOK, N_LAYERS, D_MODEL);

    /* ── Optimizer + scheduler ── */
    int n_params;
    tensor **all_params = module_parameters(&vlm->base, &n_params);
    adamw_opt *opt = adamw_create(ctx.params, all_params, n_params, LR,
                                   0.9f, 0.999f, 1e-8f, 1e-4f);

    int n_batches_approx = (imagenet_vlm_dl_total(train_dl) + BATCH_SIZE - 1) / BATCH_SIZE;
    int total_steps = n_batches_approx * MAX_EPOCHS;
    int warmup_steps = n_batches_approx * WARMUP_EPOCHS;
    lr_scheduler *sched = lr_scheduler_create(ctx.params, opt,
                                               LR_SCHEDULE_LINEAR_WARMUP_COSINE,
                                               LR, warmup_steps, total_steps,
                                               MIN_LR, 0, 0);

    /* ── Create checkpoint directory ── */
    mkdir("ckpt", 0755);

    /* ── Training ── */
    int bucket_limits[] = BUCKET_LIMITS;
    int bucket_starts[N_BUCKETS + 1];

    for (int epoch = 0; epoch < MAX_EPOCHS; epoch++) {
        imagenet_vlm_dl_shuffle(train_dl);
        imagenet_vlm_dl_bucket(train_dl, N_BUCKETS, bucket_limits, bucket_starts);

        double epoch_loss = 0.0;
        int batch_count = 0;
        struct timespec t0;
        clock_gettime(CLOCK_MONOTONIC, &t0);

        int bucket_id = 0;
        while (bucket_id < N_BUCKETS) {
            int lo = bucket_starts[bucket_id];
            int hi = bucket_starts[bucket_id + 1];
            int available = hi - lo;
            if (available <= 0) { bucket_id++; continue; }

            int take = available < BATCH_SIZE ? available : BATCH_SIZE;
            int T = BUCKET_TMAX[bucket_id];

            tensor *img = tensor_scratch(ctx.scratch, 4,
                                          (int[]){take, IMG_C, IMG_H, IMG_W}, 0);
            tensor *input_ids = tensor_zeros_data(ctx.data, 2,
                                                   (int[]){take, T});
            tensor *target_ids = tensor_zeros_data(ctx.data, 2,
                                                    (int[]){take, T});
            tensor *loss_mask = tensor_scratch(ctx.scratch, 2,
                                                (int[]){take, T}, 0);
            int text_lens[take];
            int label_ids[take];

            int got = imagenet_vlm_dl_next_batch(train_dl, img, input_ids,
                                                  target_ids, loss_mask,
                                                  text_lens, label_ids, take);
            if (got < 0) {
                fprintf(stderr, "FATAL: dataloader error at pos %ld\n",
                        (long)train_dl->pos);
                return 1;
            }
            if (got == 0) { bucket_id++; continue; }

            float grad_norm;
            tensor *loss = vision_lm_train_step_padded(ctx.scratch, vlm,
                                                         img, input_ids,
                                                         target_ids, loss_mask,
                                                         text_lens,
                                                         opt, GRAD_CLIP, &grad_norm);
            float lv = tensor_data_ptr(loss)[0];
            lr_scheduler_step(sched);
            epoch_loss += lv;
            batch_count++;

            if (batch_count % 10 == 0 || batch_count == 1) {
                struct timespec now;
                clock_gettime(CLOCK_MONOTONIC, &now);
                double elapsed = (now.tv_sec - t0.tv_sec)
                               + (now.tv_nsec - t0.tv_nsec) / 1e9;
                float cur_lr = lr_scheduler_get_lr(sched);
                double eta = (elapsed / batch_count)
                           * (n_batches_approx - batch_count);
                printf("  e%d  batch %4d/%-4d  loss %.6f  lr %.2e  gn %.4e  T=%d  %.1f/s  eta %ds\n",
                       epoch + 1, batch_count, n_batches_approx,
                       epoch_loss / batch_count,
                       cur_lr, grad_norm, T, batch_count / elapsed,
                       (int)eta);
            }
            if (batch_count % 500 == 0 && label_names && val_dl) {
                show_preds(ctx.scratch, ctx.data, vlm, val_dl,
                           (const char **)label_names);
                mem_pool_reset(ctx.scratch);
                mem_pool_reset(ctx.data);
            }

            mem_pool_reset(ctx.scratch);
            mem_pool_reset(ctx.data);

            bucket_starts[bucket_id] += got;
        }

        struct timespec t1;
        clock_gettime(CLOCK_MONOTONIC, &t1);
        double secs = (t1.tv_sec - t0.tv_sec) + (t1.tv_nsec - t0.tv_nsec) / 1e9;
        printf("  ── epoch %3d done  loss %.6f  %.1fs  %.1f batch/s\n",
               epoch + 1, epoch_loss / batch_count, secs, batch_count / secs);

        /* ── Validation ── */
        if ((epoch + 1) % 2 == 0 && val_dl && label_names) {
            int eval_n = imagenet_vlm_dl_total(val_dl) < 200 ? imagenet_vlm_dl_total(val_dl) : 200;
            float acc = eval_full_string(ctx.scratch, ctx.data, vlm, val_dl,
                                         (const char **)label_names, n_labels, eval_n);
            if (acc < 0.0f) {
                fprintf(stderr, "eval failed\n");
            } else {
                printf("  ── val full-string acc: %.3f (%d samples)\n", acc, eval_n);
            }
        }

        /* ── Checkpoint ── */
        {
            char path[128];
            time_t now = time(NULL);
            struct tm *tm = localtime(&now);
            char ts[32];
            strftime(ts, sizeof(ts), "%Y%m%dT%H%M%S", tm);
            snprintf(path, sizeof(path), "ckpt/imagenet_vlm_epoch%02d_%s.bin",
                     epoch + 1, ts);
            module_save(&vlm->base, path);
            printf("  ── saved %s\n", path);
        }
    }

    imagenet_vlm_dl_free(train_dl);
    if (val_dl) imagenet_vlm_dl_free(val_dl);
    if (label_names) {
        for (int i = 0; i < n_labels; i++) free(label_names[i]);
        free(label_names);
    }
    dnn_ctx_destroy(&ctx);
    printf("Done.\n");
    return 0;
}
