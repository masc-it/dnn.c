#include "dnn.h"
#include "transformer.h"
#include "optim.h"
#include "tokenizer.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <assert.h>
#include <time.h>

/* ── Binary dataset loader ──
 *
 * Reads promessi.bin format:
 *   Header (64 bytes): magic(4) + version(4) + vocab_size(4)
 *                       + num_sequences(4) + seq_len(4) + reserved(44)
 *   Data: num_sequences × seq_len × int32_t token IDs
 */

typedef struct {
    int   num_sequences;
    int   seq_len;
    int   vocab_size;
    int  *data;          /* [num_sequences, seq_len] flat int32 IDs */
    long  data_n;        /* num_sequences * seq_len */
} lm_dataset;

static lm_dataset load_dataset(const char *path) {
    lm_dataset ds;
    memset(&ds, 0, sizeof(ds));

    FILE *f = fopen(path, "rb");
    if (!f) { perror("fopen"); return ds; }

    unsigned char hdr[64];
    if (fread(hdr, 1, 64, f) < 64) {
        fprintf(stderr, "load_dataset: short header\n");
        fclose(f);
        return ds;
    }

    unsigned int magic = ((unsigned int)hdr[0] << 24)
                       | ((unsigned int)hdr[1] << 16)
                       | ((unsigned int)hdr[2] << 8)
                       |  (unsigned int)hdr[3];
    if (magic != TOKENIZER_DATA_MAGIC) {
        fprintf(stderr, "load_dataset: bad magic 0x%08X (expected 0x%08X)\n",
                magic, TOKENIZER_DATA_MAGIC);
        fclose(f);
        return ds;
    }

    /* read little-endian int32 fields from header */
    ds.vocab_size    = (int)hdr[8]  | ((int)hdr[9]  << 8)
                     | ((int)hdr[10] << 16) | ((int)hdr[11] << 24);
    ds.num_sequences = (int)hdr[12] | ((int)hdr[13] << 8)
                     | ((int)hdr[14] << 16) | ((int)hdr[15] << 24);
    ds.seq_len       = (int)hdr[16] | ((int)hdr[17] << 8)
                     | ((int)hdr[18] << 16) | ((int)hdr[19] << 24);

    long data_bytes = (long)ds.num_sequences * (long)ds.seq_len * 4;
    ds.data_n = (long)ds.num_sequences * (long)ds.seq_len;

    ds.data = (int *)malloc(data_bytes);
    if (!ds.data) { fprintf(stderr, "load_dataset: malloc(%ld) failed\n", data_bytes); fclose(f); return ds; }

    if ((long)fread(ds.data, 1, data_bytes, f) < data_bytes) {
        fprintf(stderr, "load_dataset: short data read\n");
        free(ds.data); ds.data = NULL;
        fclose(f);
        return ds;
    }

    fclose(f);
    return ds;
}

/* Fisher-Yates shuffle */
static void shuffle_int(int *arr, int n) {
    for (int i = n - 1; i > 0; i--) {
        int j = rand() % (i + 1);
        int t = arr[i]; arr[i] = arr[j]; arr[j] = t;
    }
}

/* ── Config ── */
#define D_MODEL        128
#define N_HEADS         4
#define D_K            32   /* D_MODEL / N_HEADS */
#define INTERMEDIATE   256  /* 2x D_MODEL */
#define N_LAYERS         4
#define VOCAB_SIZE     261
#define BATCH_SIZE      16
#define MAX_EPOCHS      10
#define LR            0.001f
#define LOG_EVERY       10

int main(void) {
    /* ── Pools ── */
    mem_pool params  = mem_pool_create(128 * 1024 * 1024);  /* model + opt state */
    mem_pool scratch = mem_pool_create((size_t)4 * 1024 * 1024 * 1024);  /* activations (B=16, N=514, d=128, 4 layers) */
    mem_pool data    = mem_pool_create(10 * 1024 * 1024);   /* batch data */
    mem_pool_set_defaults(&params, &scratch, &data);

    /* ── Load dataset ── */
    printf("Loading dataset...\n");
    lm_dataset ds = load_dataset("data/promessi.bin");
    if (!ds.data) {
        fprintf(stderr, "Failed to load data/promessi.bin\n");
        goto cleanup;
    }
    printf("  %d sequences, seq_len=%d, vocab=%d\n",
           ds.num_sequences, ds.seq_len, ds.vocab_size);
    printf("  total tokens: %ld\n", ds.data_n);

    /* verify vocab size matches */
    assert(ds.vocab_size == VOCAB_SIZE);

    /* ── Create model ── */
    printf("Creating decoder LM (%d layers, d_model=%d, n_heads=%d)...\n",
           N_LAYERS, D_MODEL, N_HEADS);

    /* set seed for reproducibility */
    srand(42);

    decoder_lm *lm = decoder_lm_create(VOCAB_SIZE, D_MODEL, N_LAYERS,
                                        N_HEADS, D_K, INTERMEDIATE);
    printf("  model created.\n");

    /* ── Collect all trainable params ── */
    tensor *all_params[256];
    int n_params = 0;

    all_params[n_params++] = lm->embedding_table;
    all_params[n_params++] = lm->norm_weight;
    all_params[n_params++] = lm->norm_bias;
    all_params[n_params++] = lm->lm_head->weight;
    all_params[n_params++] = lm->lm_head->bias;

    for (int i = 0; i < lm->n_layers; i++) {
        transformer_block *b = lm->blocks[i];
        all_params[n_params++] = b->q_proj->weight;
        all_params[n_params++] = b->q_proj->bias;
        all_params[n_params++] = b->k_proj->weight;
        all_params[n_params++] = b->k_proj->bias;
        all_params[n_params++] = b->v_proj->weight;
        all_params[n_params++] = b->v_proj->bias;
        all_params[n_params++] = b->out_proj->weight;
        all_params[n_params++] = b->out_proj->bias;
        all_params[n_params++] = b->attn_norm_weight;
        all_params[n_params++] = b->attn_norm_bias;
        all_params[n_params++] = b->ffn_norm_weight;
        all_params[n_params++] = b->ffn_norm_bias;
        all_params[n_params++] = b->ffn->gate_proj->weight;
        all_params[n_params++] = b->ffn->gate_proj->bias;
        all_params[n_params++] = b->ffn->up_proj->weight;
        all_params[n_params++] = b->ffn->up_proj->bias;
        all_params[n_params++] = b->ffn->down_proj->weight;
        all_params[n_params++] = b->ffn->down_proj->bias;
    }

    printf("  %d param groups.\n", n_params);

    /* ── Create optimizer ── */
    adamw_opt *opt = adamw_create(all_params, n_params, LR,
                                   0.9f, 0.999f, 1e-8f, 0.01f);

    /* ── Training loop ── */
    int N          = ds.seq_len;
    int n_seqs     = ds.num_sequences;
    int batch_size = BATCH_SIZE;
    int n_batches  = (n_seqs + batch_size - 1) / batch_size;

    printf("\nTraining (AdamW, lr=%.4f, batch=%d, max_epochs=%d):\n",
           LR, batch_size, MAX_EPOCHS);
    printf("  sequences=%d  seq_len=%d  batches/epoch=%d\n",
           n_seqs, N, n_batches);

    /* shuffle index buffer */
    int *indices = malloc((size_t)n_seqs * sizeof(int));
    int total_steps = 0;

    for (int epoch = 0; epoch < MAX_EPOCHS; epoch++) {
        /* shuffle sequences */
        for (int i = 0; i < n_seqs; i++) indices[i] = i;
        shuffle_int(indices, n_seqs);

        double epoch_loss = 0.0;
        int    epoch_batches = 0;
        struct timespec epoch_t0;
        clock_gettime(CLOCK_MONOTONIC, &epoch_t0);

        for (int b = 0; b < n_batches; b++) {
            int start = b * batch_size;
            int bs    = (b == n_batches - 1) ? n_seqs - start : batch_size;

            /* ── Build batch tensor ── */
            tensor *input_ids = tensor_zeros_data(2, (int[]){bs, N});
            int *id_data = (int *)input_ids->data;

            for (int i = 0; i < bs; i++) {
                int seq_idx = indices[start + i];
                int *src = ds.data + (long)seq_idx * N;
                memcpy(id_data + (long)i * N, src, (size_t)N * sizeof(int));
            }

            /* ── Train step ── */
            tensor *loss = decoder_lm_train_step(lm, input_ids, opt);
            float loss_val = ((float *)loss->data)[0];

            epoch_loss    += loss_val;
            epoch_batches++;
            total_steps++;

            /* ── Log every LOG_EVERY batches ── */
            if ((b + 1) % LOG_EVERY == 0 || b == 0 || b == n_batches - 1) {
                struct timespec now;
                clock_gettime(CLOCK_MONOTONIC, &now);
                double elapsed = (double)(now.tv_sec - epoch_t0.tv_sec)
                               + (double)(now.tv_nsec - epoch_t0.tv_nsec) / 1e9;
                double batch_s = (double)(b + 1) / elapsed;

                printf("  epoch %2d/%d  batch %4d/%d  loss %.6f  %.1f batch/s\n",
                       epoch + 1, MAX_EPOCHS, b + 1, n_batches,
                       epoch_loss / epoch_batches, batch_s);
            }

            /* free scratch + data */
            mem_pool_reset(&scratch);
            mem_pool_reset(&data);
        }

        struct timespec epoch_t1;
        clock_gettime(CLOCK_MONOTONIC, &epoch_t1);
        double epoch_sec = (double)(epoch_t1.tv_sec - epoch_t0.tv_sec)
                         + (double)(epoch_t1.tv_nsec - epoch_t0.tv_nsec) / 1e9;

        printf("  ── epoch %2d done  avg loss %.6f  %.2fs  %.1f batch/s\n",
               epoch + 1, epoch_loss / epoch_batches,
               epoch_sec, n_batches / epoch_sec);
    }

    printf("\nTraining complete.  Total steps: %d\n", total_steps);

    /* ── Cleanup ── */
    free(indices);
    free(ds.data);
    adamw_free(opt);

cleanup:
    mem_pool_destroy(&params);
    mem_pool_destroy(&scratch);
    mem_pool_destroy(&data);

    printf("Done.\n");
    return ds.data ? 0 : 1;
}
