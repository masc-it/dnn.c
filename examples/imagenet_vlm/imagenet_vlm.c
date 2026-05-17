/* ══════════════════════════════════════════════════════════════════
 *  imagenet_vlm — train VLM on ImageNet-1k
 *
 *  Treats classification as label-text generation with byte-level
 *  tokenizer, prefix-LM attention, shuffled padded batches.
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
#define WARMUP_EPOCHS     1
#define PATCH_LR_MULT    1.0f
#define VAL_CE_PROBE_EVERY 300
#define VAL_CE_PROBE_N      64
#define VAL_CE_PROBE_BS     16
#define VAL_EPOCH_CE_PROBE_N 256

/* Single padded sequence length. No bucketing: every shuffled batch uses
 * IMAGENET_MAX_TEXT_LEN, masks out pad positions, and preserves sample order
 * produced by the dataloader shuffle. */
#define TRAIN_T          IMAGENET_MAX_TEXT_LEN

/* ── Helpers ── */

static int vlm_decode_ar(struct mem_pool *scratch, struct mem_pool *data,
                          vision_lm *vlm, tensor *img,
                          int take, int got, int max_decode,
                          int tokens[][IMAGENET_MAX_TEXT_LEN]) {
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
    return seq_len;
}

static void vlm_tokens_to_str(const int tokens[], int seq_len,
                               char *out, int out_size) {
    int pn = 0;
    for (int t = 1; t < seq_len; t++) {
        int tok = tokens[t];
        if (tok == TOKENIZER_EOS_ID) break;
        if (tok < 0 || tok > 255) break;
        if (pn < out_size - 1)
            out[pn++] = (tok >= 32 && tok < 127) ? (char)tok : '?';
    }
    out[pn] = '\0';
}

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
        int seq_len = vlm_decode_ar(scratch, data, vlm, img, take, got, max_decode, tokens);

        for (int i = 0; i < got; i++) {
            char pred_name[IMAGENET_MAX_TEXT_LEN];
            vlm_tokens_to_str(tokens[i], seq_len, pred_name, sizeof(pred_name));
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
    int seq_len = vlm_decode_ar(scratch, data, vlm, img, take, got, max_decode, tokens);

    /* Decode predicted byte sequences to strings and print. */
    printf("  ── preds ──\n");
    for (int i = 0; i < got; i++) {
        char pred_name[IMAGENET_MAX_TEXT_LEN];
        vlm_tokens_to_str(tokens[i], seq_len, pred_name, sizeof(pred_name));
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
 *  Training instrumentation helpers
 * ══════════════════════════════════════════════════════════════════ */

#define MAX_NAMED_PARAMS 256
#define PARAM_NAME_MAX   256

typedef struct {
    char name[PARAM_NAME_MAX];
    tensor *param;
} named_param;

typedef enum {
    PG_PATCH = 0,
    PG_IMG_POS,
    PG_TOK_EMBED,
    PG_BLOCKS,
    PG_LM_NORM,
    PG_LM_HEAD,
    PG_OTHER,
    PG_COUNT
} param_group_id;

typedef struct {
    double grad_sq;
    double weight_sq;
    double update_sq;
    long long n_params;
} param_group_stat;

static const char *PG_NAME[PG_COUNT] = {
    "patch", "img_pos", "tok_emb", "blocks", "lm_norm", "lm_head", "other"
};

static int starts_with(const char *s, const char *prefix) {
    size_t n = strlen(prefix);
    return strncmp(s, prefix, n) == 0;
}

static param_group_id param_group_for_name(const char *name) {
    if (starts_with(name, "patch_embed.")) return PG_PATCH;
    if (strcmp(name, "image_pos") == 0) return PG_IMG_POS;
    if (starts_with(name, "lm.embed.")) return PG_TOK_EMBED;
    if (starts_with(name, "lm.blocks.")) return PG_BLOCKS;
    if (starts_with(name, "lm.norm.")) return PG_LM_NORM;
    if (starts_with(name, "lm.lm_head.")) return PG_LM_HEAD;
    return PG_OTHER;
}

static void collect_named_params_rec(module *m, const char *prefix,
                                      named_param *out, int max_out, int *n_out) {
    for (module_item *item = m->items_head; item && *n_out < max_out; item = item->next) {
        if (item->kind == MODULE_ITEM_PARAM) {
            snprintf(out[*n_out].name, sizeof(out[*n_out].name), "%s%s", prefix, item->name);
            out[*n_out].param = item->as.param;
            (*n_out)++;
        } else if (item->kind == MODULE_ITEM_CHILD) {
            char child_prefix[PARAM_NAME_MAX];
            snprintf(child_prefix, sizeof(child_prefix), "%s%s.", prefix, item->name);
            collect_named_params_rec(item->as.child, child_prefix, out, max_out, n_out);
        }
    }
}

static int collect_named_params(module *m, named_param *out, int max_out) {
    int n = 0;
    collect_named_params_rec(m, "", out, max_out, &n);
    return n;
}

static void reset_group_stats(param_group_stat st[PG_COUNT]) {
    memset(st, 0, PG_COUNT * sizeof(param_group_stat));
}

static void collect_grad_weight_stats(named_param *params, int n_params,
                                      param_group_stat st[PG_COUNT]) {
    reset_group_stats(st);
    for (int i = 0; i < n_params; i++) {
        tensor *p = params[i].param;
        int n = tensor_numel(p);
        float *wd = tensor_data_ptr(p);
        float *gd = tensor_grad(p);
        param_group_id gid = param_group_for_name(params[i].name);
        st[gid].n_params += n;
        for (int j = 0; j < n; j++) {
            double w = wd[j];
            st[gid].weight_sq += w * w;
            if (gd) {
                double g = gd[j];
                st[gid].grad_sq += g * g;
            }
        }
    }
}

static long long total_param_numel(named_param *params, int n_params) {
    long long total = 0;
    for (int i = 0; i < n_params; i++) total += tensor_numel(params[i].param);
    return total;
}

static float *snapshot_params(named_param *params, int n_params, long long *n_out) {
    long long total = total_param_numel(params, n_params);
    float *snap = malloc((size_t)total * sizeof(float));
    if (!snap) { *n_out = 0; return NULL; }
    long long off = 0;
    for (int i = 0; i < n_params; i++) {
        tensor *p = params[i].param;
        int n = tensor_numel(p);
        memcpy(snap + off, tensor_data_ptr(p), (size_t)n * sizeof(float));
        off += n;
    }
    *n_out = total;
    return snap;
}

static void collect_update_stats(named_param *params, int n_params,
                                 const float *snap,
                                 param_group_stat st[PG_COUNT]) {
    long long off = 0;
    for (int i = 0; i < n_params; i++) {
        tensor *p = params[i].param;
        int n = tensor_numel(p);
        float *wd = tensor_data_ptr(p);
        param_group_id gid = param_group_for_name(params[i].name);
        for (int j = 0; j < n; j++) {
            double d = (double)wd[j] - (double)snap[off + j];
            st[gid].update_sq += d * d;
        }
        off += n;
    }
}

static void batch_token_stats(const tensor *loss_mask, const int *text_lens, int B,
                              int *tok_out, float *mean_len_out,
                              int *min_len_out, int *max_len_out) {
    int T = loss_mask->shape[1];
    float *md = tensor_data_ptr((tensor *)loss_mask);
    int tok = 0, min_len = text_lens[0], max_len = text_lens[0];
    double sum_len = 0.0;
    for (int b = 0; b < B; b++) {
        int tl = text_lens[b];
        if (tl < min_len) min_len = tl;
        if (tl > max_len) max_len = tl;
        sum_len += tl;
        for (int t = 0; t < T; t++)
            if (md[(long)b * loss_mask->strides[0] + t * loss_mask->strides[1]] != 0.0f)
                tok++;
    }
    *tok_out = tok;
    *mean_len_out = (float)(sum_len / (double)B);
    *min_len_out = min_len;
    *max_len_out = max_len;
}

static void print_group_stat(param_group_stat st[PG_COUNT], param_group_id gid) {
    double w = sqrt(st[gid].weight_sq);
    double u = sqrt(st[gid].update_sq);
    double uw = (w > 0.0) ? u / w : 0.0;
    double grms = (st[gid].n_params > 0)
                ? sqrt(st[gid].grad_sq / (double)st[gid].n_params)
                : 0.0;
    printf("%s grms=%.2e u/w=%.2e", PG_NAME[gid], grms, uw);
}

static void print_key_group_stats(param_group_stat st[PG_COUNT]) {
    printf("       grp ");
    print_group_stat(st, PG_PATCH);
    printf(" | ");
    print_group_stat(st, PG_IMG_POS);
    printf(" | ");
    print_group_stat(st, PG_TOK_EMBED);
    printf(" | ");
    print_group_stat(st, PG_BLOCKS);
    printf("\n");
}

static int count_mask_tokens(const tensor *loss_mask) {
    int B = loss_mask->shape[0];
    int T = loss_mask->shape[1];
    float *md = tensor_data_ptr((tensor *)loss_mask);
    int tok = 0;
    for (int b = 0; b < B; b++)
        for (int t = 0; t < T; t++)
            if (md[(long)b * loss_mask->strides[0] + t * loss_mask->strides[1]] != 0.0f)
                tok++;
    return tok;
}

static int eval_real_blank_ce(struct mem_pool *scratch, struct mem_pool *data,
                              vision_lm *vlm, imagenet_vlm_dl *dl,
                              int max_n, float *real_ce_out,
                              float *blank_ce_out) {
    if (!dl || max_n <= 0) return -1;

    long saved_pos = dl->pos;
    imagenet_vlm_dl_reset(dl);

    double real_sum = 0.0;
    double blank_sum = 0.0;
    int total_tok = 0;
    int total = 0;

    dnn_grad_ctx ng = dnn_no_grad_enter();
    while (total < max_n) {
        int take = max_n - total < VAL_CE_PROBE_BS ? max_n - total : VAL_CE_PROBE_BS;
        tensor *img = tensor_zeros_data(data, 4,
                                        (int[]){take, IMG_C, IMG_H, IMG_W});
        tensor *inp = tensor_zeros_data(data, 2, (int[]){take, IMAGENET_MAX_TEXT_LEN});
        tensor *tgt = tensor_zeros_data(data, 2, (int[]){take, IMAGENET_MAX_TEXT_LEN});
        tensor *msk = tensor_zeros_data(data, 2, (int[]){take, IMAGENET_MAX_TEXT_LEN});
        int tl[take], lbl[take];

        int got = imagenet_vlm_dl_next_batch(dl, img, inp, tgt, msk, tl, lbl, take);
        (void)lbl;
        if (got < 0) {
            mem_pool_reset(scratch);
            mem_pool_reset(data);
            dnn_no_grad_exit(ng);
            dl->pos = saved_pos;
            return -1;
        }
        if (got == 0) {
            mem_pool_reset(scratch);
            mem_pool_reset(data);
            break;
        }
        for (int i = got; i < take; i++) tl[i] = 0;

        int tok = count_mask_tokens(msk);
        if (tok <= 0) {
            total += got;
            mem_pool_reset(scratch);
            mem_pool_reset(data);
            continue;
        }

        tensor *loss_real = vision_lm_loss_padded(scratch, vlm, img, inp, tgt, msk, tl);
        float real_lv = tensor_data_ptr(loss_real)[0];
        mem_pool_reset(scratch);

        tensor *blank = tensor_zeros_data(data, 4,
                                          (int[]){take, IMG_C, IMG_H, IMG_W});
        tensor *loss_blank = vision_lm_loss_padded(scratch, vlm, blank, inp, tgt, msk, tl);
        float blank_lv = tensor_data_ptr(loss_blank)[0];

        real_sum += (double)real_lv * (double)tok;
        blank_sum += (double)blank_lv * (double)tok;
        total_tok += tok;
        total += got;

        mem_pool_reset(scratch);
        mem_pool_reset(data);
    }
    dnn_no_grad_exit(ng);
    dl->pos = saved_pos;

    if (total_tok <= 0) return -1;
    *real_ce_out = (float)(real_sum / (double)total_tok);
    *blank_ce_out = (float)(blank_sum / (double)total_tok);
    return total;
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
    dnn_ctx_init(&ctx, 256*1024*1024, (size_t)4096*1024*1024, 256*1024*1024);

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
    if (vlm->image_pos)
        printf("Image pos: enabled\n");

    /* ── Optimizer + scheduler ── */
    int n_params;
    tensor **all_params = module_parameters(&vlm->base, &n_params);
    adamw_opt *opt = adamw_create(ctx.params, all_params, n_params, LR,
                                   0.9f, 0.999f, 1e-8f, 1e-4f);

    named_param named_params[MAX_NAMED_PARAMS];
    int n_named_params = collect_named_params(&vlm->base, named_params, MAX_NAMED_PARAMS);
    if (n_named_params >= MAX_NAMED_PARAMS)
        printf("WARNING: instrumentation param list truncated at %d\n", MAX_NAMED_PARAMS);
    else
        printf("Instrumentation: %d named params\n", n_named_params);

    float *lr_mults = malloc((size_t)n_params * sizeof(float));
    if (!lr_mults) { fprintf(stderr, "Failed to allocate LR multipliers\n"); return 1; }
    for (int i = 0; i < n_params; i++) lr_mults[i] = 1.0f;
    tensor *patch_w = module_find_param(&vlm->base, "patch_embed.weight");
    tensor *patch_b = module_find_param(&vlm->base, "patch_embed.bias");
    int patch_mult_count = 0;
    for (int i = 0; i < n_params; i++) {
        if (all_params[i] == patch_w || all_params[i] == patch_b) {
            lr_mults[i] = PATCH_LR_MULT;
            patch_mult_count++;
        }
    }
    printf("LR multipliers: patch_embed x%.1f (%d tensors)\n", PATCH_LR_MULT, patch_mult_count);

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
    float best_eval_loss = INFINITY;
    for (int epoch = 0; epoch < MAX_EPOCHS; epoch++) {
        vlm->use_image_pos = (vlm->image_pos != NULL);
        imagenet_vlm_dl_shuffle(train_dl);

        double epoch_loss = 0.0;
        int batch_count = 0;
        struct timespec t0;
        clock_gettime(CLOCK_MONOTONIC, &t0);

        while (1) {
            int remaining = (int)imagenet_vlm_dl_remaining(train_dl);
            int take = remaining < BATCH_SIZE ? remaining : BATCH_SIZE;
            if (take <= 0) break;

            tensor *img = tensor_scratch(ctx.scratch, 4,
                                          (int[]){take, IMG_C, IMG_H, IMG_W}, 0);
            tensor *input_ids = tensor_zeros_data(ctx.data, 2,
                                                   (int[]){take, TRAIN_T});
            tensor *target_ids = tensor_zeros_data(ctx.data, 2,
                                                    (int[]){take, TRAIN_T});
            tensor *loss_mask = tensor_scratch(ctx.scratch, 2,
                                                (int[]){take, TRAIN_T}, 0);
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
            if (got == 0) break;

            int next_batch = batch_count + 1;
            int log_now = (next_batch % 10 == 0 || next_batch == 1);
            int batch_tokens = 0, min_len = 0, max_len = 0;
            float mean_len = 0.0f;
            param_group_stat group_stats[PG_COUNT];
            long long snap_n = 0;
            float *snap = NULL;

            adamw_zero_grad(opt);
            tensor *loss = vision_lm_loss_padded(ctx.scratch, vlm,
                                                  img, input_ids,
                                                  target_ids, loss_mask,
                                                  text_lens);
            dnn_backward(ctx.scratch, loss);

            if (log_now) {
                batch_token_stats(loss_mask, text_lens, got,
                                  &batch_tokens, &mean_len, &min_len, &max_len);
                collect_grad_weight_stats(named_params, n_named_params, group_stats);
                snap = snapshot_params(named_params, n_named_params, &snap_n);
            }

            float gn = 0.0f;
            if (GRAD_CLIP > 0.0f)
                gn = clip_grad_norm(opt->params, opt->n_params, GRAD_CLIP);
            else
                gn = grad_norm(opt->params, opt->n_params);

            adamw_step_with_lr_multipliers(opt, lr_mults);
            if (log_now && snap) {
                (void)snap_n;
                collect_update_stats(named_params, n_named_params, snap, group_stats);
                free(snap);
            }

            float lv = tensor_data_ptr(loss)[0];
            lr_scheduler_step(sched);
            epoch_loss += lv;
            batch_count = next_batch;

            if (log_now) {
                struct timespec now;
                clock_gettime(CLOCK_MONOTONIC, &now);
                double elapsed = (now.tv_sec - t0.tv_sec)
                               + (now.tv_nsec - t0.tv_nsec) / 1e9;
                float cur_lr = lr_scheduler_get_lr(sched);
                double eta = (elapsed / batch_count)
                           * (n_batches_approx - batch_count);
                printf("e%d  batch %4d/%-4d  loss %.6f  avg %.6f  lr %.2e  gn %.4e  tok=%d  len=%.1f/%d-%d  %.1f/s  eta %ds\n",
                       epoch + 1, batch_count, n_batches_approx,
                       lv, epoch_loss / batch_count,
                       cur_lr, gn,
                       batch_tokens, mean_len, min_len, max_len,
                       batch_count / elapsed, (int)eta);
                print_key_group_stats(group_stats);
            }
            
            if (batch_count % 500 == 0 && label_names && val_dl) {
                show_preds(ctx.scratch, ctx.data, vlm, val_dl,
                           (const char **)label_names);
                mem_pool_reset(ctx.scratch);
                mem_pool_reset(ctx.data);

                float real_ce = 0.0f, blank_ce = 0.0f;
                int n_eval = eval_real_blank_ce(
                    ctx.scratch, ctx.data, vlm, val_dl,
                    VAL_CE_PROBE_N, &real_ce, &blank_ce);
                if (n_eval > 0) {
                    printf("val_ce n=%d real=%.6f blank=%.6f gap=%.6f\n",
                           n_eval, real_ce, blank_ce, blank_ce - real_ce);
                } else {
                    printf("val_ce failed\n");
                }
            }

            mem_pool_reset(ctx.scratch);
            mem_pool_reset(ctx.data);
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

        /* ── Eval loss ── */
        float epoch_ce_real = INFINITY, epoch_ce_blank = INFINITY;
        int epoch_eval_n = eval_real_blank_ce(ctx.scratch, ctx.data, vlm, val_dl,
                                              VAL_EPOCH_CE_PROBE_N,
                                              &epoch_ce_real, &epoch_ce_blank);
        if (epoch_eval_n > 0) {
            printf("  ── epoch-end val_ce real=%.6f blank=%.6f gap=%.6f (n=%d)\n",
                   epoch_ce_real, epoch_ce_blank,
                   epoch_ce_blank - epoch_ce_real, epoch_eval_n);
        } else if (!val_dl) {
            printf("  ── epoch-end val_ce skipped (no val set)\n");
        } else {
            printf("  ── epoch-end val_ce failed\n");
        }

        /* ── Checkpoint (conditional: eval loss improved) ── */
        if (epoch_eval_n > 0 && epoch_ce_real < best_eval_loss) {
            best_eval_loss = epoch_ce_real;
            char path[128];
            snprintf(path, sizeof(path), "ckpt/imagenet_vlm_best.bin");
            module_save(&vlm->base, path);
            printf("  ── best checkpoint saved (val_ce=%.6f) <%s>\n", epoch_ce_real, path);
        }

        /* ── Periodic checkpoint (unconditional timestamped) ── */
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

    free(lr_mults);
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
