#ifndef PROMESSI_EXAMPLE_H
#define PROMESSI_EXAMPLE_H

#include "dnn.h"       /* transformer.h, tokenizer.h, ops.h, etc. */
#include "gpt.h"       /* decoder_lm struct + functions */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <assert.h>
#include <time.h>

/* ── Config ── */
#define D_MODEL        256
#define N_HEADS         4
#define D_K            64   /* D_MODEL / N_HEADS */
#define INTERMEDIATE   512  /* 2x D_MODEL */
#define N_LAYERS         2
#define VOCAB_SIZE     261
#define BATCH_SIZE      16
#define MAX_EPOCHS      10
#define LR            8e-4f
#define MIN_LR            6e-5f
#define OVERFIT         1    /* 1 = train on 10 batches only for overfit test */
#if OVERFIT
#  undef  MAX_EPOCHS
#  define MAX_EPOCHS 1000
#endif
#define LOG_EVERY       10
#define GEN_EVERY       30
#define GEN_NEW_TOKENS  64

/* ── Binary dataset ── */
typedef struct {
    int   num_sequences;
    int   seq_len;
    int   vocab_size;
    int  *data;          /* [num_sequences, seq_len] flat int32 IDs */
    long  data_n;        /* num_sequences * seq_len */
} lm_dataset;

lm_dataset load_dataset(const char *path);
void       shuffle_int(int *arr, int n);

#endif /* PROMESSI_EXAMPLE_H */
