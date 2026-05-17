#ifndef DNN_IMAGENET_VLM_H
#define DNN_IMAGENET_VLM_H

#include <stdint.h>
#include "tensor.h"
#include "tokenizer.h"

/* ══════════════════════════════════════════════════════════════════
 *  Constants
 * ══════════════════════════════════════════════════════════════════ */

#define IMAGENET_MAGIC         0x494D474E   /* "IMGN" */
#define IMAGENET_INDEX_MAGIC   0x58444E49   /* "INDX" */
#define IMAGENET_CLASSES       1000
#define IMAGENET_PAD_ID        TOKENIZER_PAD_ID   /* 259 */

/* Max (stored_text_len + 1) = BOS + raw_label_bytes + EOS.
 * Longest synset name is 121 bytes; +EOS = 122 stored; +BOS = 123.
 * Bucket 4 T=128 covers all. */
#define IMAGENET_MAX_TEXT_LEN  128

#define IMAGENET_MEAN          {0.485f, 0.456f, 0.406f}
#define IMAGENET_STD           {0.229f, 0.224f, 0.225f}

/* ══════════════════════════════════════════════════════════════════
 *  On-disk types
 * ══════════════════════════════════════════════════════════════════ */

typedef struct __attribute__((packed)) {
    int32_t magic;
    int32_t version;
    int32_t H, W, C;
    int32_t num_samples;
    int32_t shard_idx;
    int32_t num_shards;
    int8_t  reserved[32];
} imagenet_shard_header;

typedef struct __attribute__((packed)) {
    uint32_t shard;       /* shard_idx this sample belongs to */
    uint32_t local;       /* sample index within shard (0-based) */
    uint64_t offset;      /* byte offset from start of shard body (after header) */
    uint16_t text_len;    /* stored text_len (bytes + EOS, no BOS) */
    uint16_t label_id;    /* class index 0–999 */
} imagenet_idx_entry;     /* 20 bytes */

/* ══════════════════════════════════════════════════════════════════
 *  DataLoader struct
 * ══════════════════════════════════════════════════════════════════ */

typedef struct imagenet_vlm_dl {
    /* Image geometry (from shard headers) */
    int H, W, C;
    int sample_pixel_bytes;     /* H * W * C */

    /* Shard info */
    int      num_shards;
    char     shard_pattern[256]; /* sprintf pattern for .bin paths */

    /* Per-shard buffer (malloc, ~1 GB) */
    uint8_t *buffer;
    long     buffer_capacity;
    long     buffer_bytes;
    int      current_shard;      /* -1 = none loaded */

    /* Split index (mmap'd) */
    imagenet_idx_entry *idx;
    size_t              idx_bytes;
    int    total_samples;

    /* Per-shard index view and metadata */
    int    shard_num_samples;
    const imagenet_idx_entry *shard_entries;
    int    current_shard_version;  /* version from shard header, for format compat */

    /* Physical shard → idx range (immutable after create) */
    int *shard_start;        /* [num_shards + 1], idx index of first sample */
    int *shard_count;        /* [num_shards], samples per shard */

    /* Permutation */
    int  *shuffle_order;     /* [total_samples], global idx indices in visit order */
    int  *shard_bounds;      /* [num_shards + 1], mutable visit-order ranges */
    int  *shard_visit;       /* [num_shards], shuffled shard order each epoch */
    long  pos;
    int   shuffle;

    /* Bucket state */
    int   n_buckets;
    int  *bucket_starts;     /* [n_buckets + 1] */

    /* RNG */
    uint64_t rng_state;

    /* Normalisation */
    float mean[3];
    float std[3];
} imagenet_vlm_dl;

/* ══════════════════════════════════════════════════════════════════
 *  Lifecycle
 * ══════════════════════════════════════════════════════════════════ */

imagenet_vlm_dl *imagenet_vlm_dl_create(const char *split,
                                         const char *data_dir,
                                         int shuffle,
                                         unsigned long seed);
void  imagenet_vlm_dl_free(imagenet_vlm_dl *dl);
int   imagenet_vlm_dl_reset(imagenet_vlm_dl *dl);

/* ══════════════════════════════════════════════════════════════════
 *  Epoch lifecycle
 * ══════════════════════════════════════════════════════════════════ */

void imagenet_vlm_dl_shuffle(imagenet_vlm_dl *dl);
void imagenet_vlm_dl_bucket(imagenet_vlm_dl *dl,
                             int n_buckets,
                             const int *bucket_limits,
                             int *bucket_starts);

/* ══════════════════════════════════════════════════════════════════
 *  Batch construction
 * ══════════════════════════════════════════════════════════════════ */

/* Fills batch tensors from next samples in shuffle_order[].
 *
 *   img        — [bs, C, H, W] float32 NCHW (scratch pool)
 *   input_ids  — [bs, T_batch] int32 (data pool, pre-filled zeros)
 *   target_ids — [bs, T_batch] int32 (data pool, pre-filled zeros)
 *   loss_mask  — [bs, T_batch] float32 (scratch pool)
 *   text_lens  — [bs] int output (caller stack)
 *   label_ids  — [bs] int output (caller stack, may be NULL)
 *   bs         — requested batch size
 *
 *   T_batch is second dimension of input_ids/target_ids/loss_mask.
 *   Caller normally allocates IMAGENET_MAX_TEXT_LEN for fully padded,
 *   non-bucketed shuffled batches. Shorter T_batch is allowed only if every
 *   sampled label fits; otherwise returns -1.
 *
 *   Returns N samples written (0 < N <= bs).
 *   Returns 0 when epoch done (pos >= total_samples).
 *   Returns -1 on fatal error (corrupt shard/caller bug).  Abort on -1.
 */
int imagenet_vlm_dl_next_batch(imagenet_vlm_dl *dl,
                                tensor *img,
                                tensor *input_ids,
                                tensor *target_ids,
                                tensor *loss_mask,
                                int *text_lens,
                                int *label_ids,
                                int bs);

/* ══════════════════════════════════════════════════════════════════
 *  Accessors
 * ══════════════════════════════════════════════════════════════════ */

static inline int  imagenet_vlm_dl_total(const imagenet_vlm_dl *dl) { return dl->total_samples; }
static inline long imagenet_vlm_dl_remaining(const imagenet_vlm_dl *dl) { return dl->total_samples - dl->pos; }

#endif /* DNN_IMAGENET_VLM_H */
