#include "imagenet_vlm.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <dirent.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <unistd.h>
#include <sys/mman.h>

/* ══════════════════════════════════════════════════════════════════
 *  Internal helpers
 * ══════════════════════════════════════════════════════════════════ */

static inline uint64_t xorshift64(uint64_t *state) {
    uint64_t x = *state;
    x ^= x << 13; x ^= x >> 7; x ^= x << 17;
    *state = x;
    return x;
}

static void fy_shuffle(int *arr, int lo, int hi, uint64_t *rng) {
    if (hi - lo < 2) return;
    for (int i = hi - 1; i > lo; i--) {
        int j = lo + (int)(xorshift64(rng) % (unsigned)(i - lo + 1));
        int t = arr[i]; arr[i] = arr[j]; arr[j] = t;
    }
}

static void convert_pixels(float *dst, const uint8_t *src,
                            int H, int W, int C,
                            const float *mean, const float *std,
                            int flip) {
    for (int h = 0; h < H; h++) {
        for (int w = 0; w < W; w++) {
            int ws = flip ? (W - 1 - w) : w;
            const uint8_t *pixel = src + (h * (long)W + ws) * C;
            float *base = dst + h * (long)W + w;
            for (int c = 0; c < C; c++) {
                float val = pixel[c] * (1.0f / 255.0f);
                base[c * H * W] = (val - mean[c]) / std[c];
            }
        }
    }
}

static int load_shard(imagenet_vlm_dl *dl, int shard_idx) {
    if (shard_idx < 0 || shard_idx >= dl->num_shards) return -1;
    if (shard_idx == dl->current_shard) return 0;

    char path[256];
    snprintf(path, sizeof(path), dl->shard_pattern, shard_idx + 1);

    FILE *f = fopen(path, "rb");
    if (!f) return -1;

    imagenet_shard_header hdr;
    if (fread(&hdr, sizeof(hdr), 1, f) != 1) { fclose(f); return -1; }

    if (hdr.magic != IMAGENET_MAGIC
        || hdr.H != dl->H || hdr.W != dl->W || hdr.C != dl->C
        || hdr.shard_idx != (int32_t)shard_idx
        || hdr.num_shards != (int32_t)dl->num_shards)
    {
        fclose(f); return -1;
    }

    if (fseeko(f, 0, SEEK_END) != 0) { fclose(f); return -1; }
    long body_bytes = ftello(f) - (long)sizeof(hdr);
    if (body_bytes < 0 || body_bytes > dl->buffer_capacity) { fclose(f); return -1; }
    fseeko(f, (long)sizeof(hdr), SEEK_SET);

    if ((long)fread(dl->buffer, 1, (size_t)body_bytes, f) < body_bytes) {
        fclose(f); return -1;
    }
    fclose(f);

    dl->current_shard = shard_idx;
    dl->buffer_bytes = body_bytes;
    dl->shard_num_samples = (int)hdr.num_samples;
    dl->shard_entries = dl->idx + dl->shard_start[shard_idx];
    dl->current_shard_version = (int)hdr.version;

    return 0;
}

/* ══════════════════════════════════════════════════════════════════
 *  create / free / reset
 * ══════════════════════════════════════════════════════════════════ */

imagenet_vlm_dl *imagenet_vlm_dl_create(const char *split,
                                         const char *data_dir,
                                         int shuffle,
                                         unsigned long seed) {
    imagenet_vlm_dl *dl = calloc(1, sizeof(*dl));
    if (!dl) return NULL;

    /* ── Discover shards via directory scan ── */
    DIR *dir = opendir(data_dir);
    if (!dir) { free(dl); return NULL; }

    int max_idx = -1, common_n = -1;
    struct dirent *entry;
    char prefix[32];
    while ((entry = readdir(dir)) != NULL) {
        int idx, n;
        if (sscanf(entry->d_name, "%31[^-]-%d-of-%d.bin", prefix, &idx, &n) == 3
            && strcmp(prefix, split) == 0)
        {
            if (idx > max_idx) max_idx = idx;
            if (common_n == -1) common_n = n;
            else if (n != common_n) { closedir(dir); free(dl); return NULL; }
        }
    }
    closedir(dir);

    if (max_idx < 1 || common_n != max_idx) { free(dl); return NULL; }
    dl->num_shards = max_idx;

    snprintf(dl->shard_pattern, sizeof(dl->shard_pattern),
             "%s/%s-%%05d-of-%05d.bin", data_dir, split, dl->num_shards);

    /* ── Validate all shard headers ── */
    int header_H = 0, header_W = 0, header_C = 0;
    for (int s = 0; s < dl->num_shards; s++) {
        char path[256];
        snprintf(path, sizeof(path), dl->shard_pattern, s + 1);
        FILE *f = fopen(path, "rb");
        if (!f) { imagenet_vlm_dl_free(dl); return NULL; }
        imagenet_shard_header hdr;
        if (fread(&hdr, sizeof(hdr), 1, f) != 1
            || hdr.magic != IMAGENET_MAGIC
            || (hdr.version != 1 && hdr.version != 2)
            || hdr.shard_idx != (int32_t)s
            || hdr.num_shards != (int32_t)max_idx
            || hdr.num_samples <= 0)
        {
            fclose(f); imagenet_vlm_dl_free(dl); return NULL;
        }
        fclose(f);
        if (s == 0) {
            header_H = hdr.H; header_W = hdr.W; header_C = hdr.C;
        } else if (hdr.H != header_H || hdr.W != header_W || hdr.C != header_C) {
            imagenet_vlm_dl_free(dl); return NULL;
        }
    }
    dl->H = header_H; dl->W = header_W; dl->C = header_C;
    dl->sample_pixel_bytes = header_H * header_W * header_C;

    /* ── mmap split index ── */
    {
        char idx_path[256];
        snprintf(idx_path, sizeof(idx_path), "%s/%s.idx", data_dir, split);
        int fd = open(idx_path, O_RDONLY);
        if (fd < 0) { imagenet_vlm_dl_free(dl); return NULL; }
        struct stat st;
        if (fstat(fd, &st) != 0) { close(fd); imagenet_vlm_dl_free(dl); return NULL; }
        dl->idx_bytes = (size_t)st.st_size;
        if (dl->idx_bytes < 16) { close(fd); imagenet_vlm_dl_free(dl); return NULL; }

        uint8_t *m = mmap(NULL, dl->idx_bytes, PROT_READ, MAP_PRIVATE, fd, 0);
        close(fd);
        if (m == MAP_FAILED) { imagenet_vlm_dl_free(dl); return NULL; }

        uint32_t idx_magic = *(uint32_t *)(m + 0);
        uint32_t idx_ver   = *(uint32_t *)(m + 4);
        uint64_t idx_num   = *(uint64_t *)(m + 8);
        if (idx_magic != IMAGENET_INDEX_MAGIC || idx_ver != 1 || idx_num == 0) {
            munmap(m, dl->idx_bytes); imagenet_vlm_dl_free(dl); return NULL;
        }
        if (dl->idx_bytes != 16 + (size_t)idx_num * sizeof(imagenet_idx_entry)) {
            munmap(m, dl->idx_bytes); imagenet_vlm_dl_free(dl); return NULL;
        }
        dl->total_samples = (int)idx_num;
        dl->idx = (imagenet_idx_entry *)(m + 16);
    }

    /* ── Validate idx entries ── */
    {
        int prev_s = -1, prev_l = -1;
        for (int i = 0; i < dl->total_samples; i++) {
            uint32_t s = dl->idx[i].shard;
            uint32_t l = dl->idx[i].local;
            uint16_t lbl = dl->idx[i].label_id;
            if ((int)s >= dl->num_shards || lbl >= IMAGENET_CLASSES) {
                munmap((void *)((uint8_t *)dl->idx - 16), dl->idx_bytes);
                dl->idx = NULL; imagenet_vlm_dl_free(dl); return NULL;
            }
            int ok = ((int)s > prev_s && (int)l == 0)
                  || ((int)s == prev_s && (int)l == prev_l + 1);
            if (!ok) {
                munmap((void *)((uint8_t *)dl->idx - 16), dl->idx_bytes);
                dl->idx = NULL; imagenet_vlm_dl_free(dl); return NULL;
            }
            prev_s = (int)s; prev_l = (int)l;
        }
    }

    /* ── Allocate permutation arrays ── */
    dl->shard_start = malloc((size_t)(dl->num_shards + 1) * sizeof(int));
    dl->shard_count = malloc((size_t)dl->num_shards * sizeof(int));
    dl->shard_bounds = malloc((size_t)(dl->num_shards + 1) * sizeof(int));
    dl->shard_visit  = malloc((size_t)dl->num_shards * sizeof(int));
    if (!dl->shard_start || !dl->shard_count || !dl->shard_bounds || !dl->shard_visit) {
        imagenet_vlm_dl_free(dl); return NULL;
    }

    dl->shard_start[0] = 0;
    int cur_s = 0;
    for (int i = 0; i < dl->total_samples; i++) {
        while ((int)dl->idx[i].shard > cur_s) {
            dl->shard_count[cur_s] = i - dl->shard_start[cur_s];
            cur_s++;
            dl->shard_start[cur_s] = i;
        }
    }
    while (cur_s < dl->num_shards) {
        dl->shard_count[cur_s] = dl->total_samples - dl->shard_start[cur_s];
        cur_s++;
        dl->shard_start[cur_s] = dl->total_samples;
    }

    for (int s = 0; s < dl->num_shards; s++) {
        dl->shard_visit[s] = s;
        dl->shard_bounds[s] = dl->shard_start[s];
    }
    dl->shard_bounds[dl->num_shards] = dl->total_samples;

    dl->shuffle_order = malloc((size_t)dl->total_samples * sizeof(int));
    if (!dl->shuffle_order) { imagenet_vlm_dl_free(dl); return NULL; }
    for (int i = 0; i < dl->total_samples; i++)
        dl->shuffle_order[i] = i;

    /* ── Validate idx count per shard matches shard headers ── */
    for (int s = 0; s < dl->num_shards; s++) {
        char path[256];
        snprintf(path, sizeof(path), dl->shard_pattern, s + 1);
        FILE *f = fopen(path, "rb");
        if (!f) { imagenet_vlm_dl_free(dl); return NULL; }
        imagenet_shard_header hdr;
        if (fread(&hdr, sizeof(hdr), 1, f) != 1 || (long)hdr.num_samples != dl->shard_count[s]) {
            fclose(f); imagenet_vlm_dl_free(dl); return NULL;
        }
        fclose(f);
    }

    /* ── Allocate 1 GB buffer ── */
    dl->buffer_capacity = (long)1024 * 1024 * 1024;
    dl->buffer = malloc((size_t)dl->buffer_capacity);
    if (!dl->buffer) { imagenet_vlm_dl_free(dl); return NULL; }
    dl->current_shard = -1;

    dl->pos = 0;
    dl->shuffle = shuffle;
    dl->rng_state = (uint64_t)seed;
    dl->n_buckets = 0;
    dl->bucket_starts = NULL;

    float m[] = IMAGENET_MEAN;
    float s_[] = IMAGENET_STD;
    memcpy(dl->mean, m, sizeof(m));
    memcpy(dl->std, s_, sizeof(s_));

    return dl;
}

void imagenet_vlm_dl_free(imagenet_vlm_dl *dl) {
    if (!dl) return;
    free(dl->buffer);
    free(dl->shuffle_order);
    free(dl->shard_start);
    free(dl->shard_count);
    free(dl->shard_bounds);
    free(dl->shard_visit);
    free(dl->bucket_starts);
    if (dl->idx) {
        uint8_t *base = (uint8_t *)dl->idx - 16;
        munmap(base, dl->idx_bytes);
    }
    free(dl);
}

int imagenet_vlm_dl_reset(imagenet_vlm_dl *dl) {
    dl->pos = 0;
    return 0;
}

/* ══════════════════════════════════════════════════════════════════
 *  Shuffle
 * ══════════════════════════════════════════════════════════════════ */

void imagenet_vlm_dl_shuffle(imagenet_vlm_dl *dl) {
    dl->pos = 0;
    if (!dl->shuffle) return;

    int S = dl->num_shards;
    int N = dl->total_samples;

    fy_shuffle(dl->shard_visit, 0, S, &dl->rng_state);

    int *scratch = malloc((size_t)N * sizeof(int));
    if (!scratch) return;

    int pos = 0;
    for (int si = 0; si < S; si++) {
        int s = dl->shard_visit[si];
        int lo = dl->shard_start[s];
        int cnt = dl->shard_count[s];
        for (int j = 0; j < cnt; j++)
            scratch[pos + j] = lo + j;
        fy_shuffle(scratch, pos, pos + cnt, &dl->rng_state);
        pos += cnt;
    }

    memcpy(dl->shuffle_order, scratch, (size_t)N * sizeof(int));

    pos = 0;
    for (int si = 0; si < S; si++) {
        dl->shard_bounds[si] = pos;
        pos += dl->shard_count[dl->shard_visit[si]];
    }
    dl->shard_bounds[S] = N;

    free(scratch);
}

/* ══════════════════════════════════════════════════════════════════
 *  Bucket
 * ══════════════════════════════════════════════════════════════════ */

void imagenet_vlm_dl_bucket(imagenet_vlm_dl *dl,
                             int n_buckets,
                             const int *bucket_limits,
                             int *bucket_starts) {
    dl->pos = 0;
    if (n_buckets <= 0) return;

    int N = dl->total_samples;

    int *counts = calloc((size_t)(n_buckets + 1), sizeof(int));
    if (!counts) return;

    for (int i = 0; i < N; i++) {
        int gi = dl->shuffle_order[i];
        int tlen = dl->idx[gi].text_len + 1;
        int bi = 0;
        while (bi < n_buckets && tlen >= bucket_limits[bi]) bi++;
        if (bi >= n_buckets) bi = n_buckets - 1;
        counts[bi]++;
    }

    bucket_starts[0] = 0;
    for (int bi = 0; bi < n_buckets; bi++)
        bucket_starts[bi + 1] = bucket_starts[bi] + counts[bi];

    int *scratch = malloc((size_t)N * sizeof(int));
    int *cur = malloc((size_t)n_buckets * sizeof(int));
    if (!scratch || !cur) { free(counts); free(scratch); free(cur); return; }

    memcpy(scratch, dl->shuffle_order, (size_t)N * sizeof(int));
    for (int bi = 0; bi < n_buckets; bi++) cur[bi] = bucket_starts[bi];

    for (int i = 0; i < N; i++) {
        int gi = scratch[i];
        int tlen = dl->idx[gi].text_len + 1;
        int bi = 0;
        while (bi < n_buckets && tlen >= bucket_limits[bi]) bi++;
        if (bi >= n_buckets) bi = n_buckets - 1;
        dl->shuffle_order[cur[bi]++] = gi;
    }

    /* Sort each bucket by shard for I/O locality */
    for (int bi = 0; bi < n_buckets; bi++) {
        int lo = bucket_starts[bi];
        int hi = bucket_starts[bi + 1];
        int cnt = hi - lo;
        if (cnt < 2) continue;

        memcpy(scratch, dl->shuffle_order + lo, (size_t)cnt * sizeof(int));

        int *shard_cnt = calloc((size_t)dl->num_shards, sizeof(int));
        if (!shard_cnt) continue;
        for (int i = 0; i < cnt; i++) {
            int gi = scratch[i];
            shard_cnt[dl->idx[gi].shard]++;
        }

        int *shard_pos = malloc((size_t)(dl->num_shards + 1) * sizeof(int));
        if (!shard_pos) { free(shard_cnt); continue; }
        shard_pos[0] = lo;
        for (int s = 0; s < dl->num_shards; s++)
            shard_pos[s + 1] = shard_pos[s] + shard_cnt[s];

        for (int i = 0; i < cnt; i++) {
            int gi = scratch[i];
            int s = dl->idx[gi].shard;
            dl->shuffle_order[shard_pos[s]++] = gi;
        }

        free(shard_cnt);
        free(shard_pos);
    }

    dl->n_buckets = n_buckets;
    free(dl->bucket_starts);
    dl->bucket_starts = malloc((size_t)(n_buckets + 1) * sizeof(int));
    if (dl->bucket_starts)
        memcpy(dl->bucket_starts, bucket_starts, (size_t)(n_buckets + 1) * sizeof(int));

    free(counts);
    free(scratch);
    free(cur);
}

/* ══════════════════════════════════════════════════════════════════
 *  next_batch
 * ══════════════════════════════════════════════════════════════════ */

int imagenet_vlm_dl_next_batch(imagenet_vlm_dl *dl,
                                tensor *img,
                                tensor *input_ids,
                                tensor *target_ids,
                                tensor *loss_mask,
                                int *text_lens,
                                int *label_ids,
                                int bs) {
    if (dl->pos >= dl->total_samples) return 0;

    int bs_actual = 0;
    int H = dl->H, W = dl->W, C = dl->C;
    int px_bytes = dl->sample_pixel_bytes;
    int T_max = input_ids->shape[1];

    for (; bs_actual < bs && dl->pos < dl->total_samples; bs_actual++, dl->pos++) {
        int gi = dl->shuffle_order[dl->pos];
        const imagenet_idx_entry *e = &dl->idx[gi];

        /* Fatal on any corrupt / caller bug — always -1, never partial batch */
        int stored_len = e->text_len;
        if ((int)e->text_len + 1 > T_max) return -1;

        if ((int)e->shard != dl->current_shard) {
            if (load_shard(dl, (int)e->shard) != 0) return -1;
        }

        long sample_off = (long)e->offset;
        if (sample_off < 0) return -1;

        const int32_t *sample_hdr = (const int32_t *)(dl->buffer + sample_off);
        const int32_t *txt = sample_hdr + 2;

        /* Skip caption field (version >= 2): caption_len(int32) follows text_ids. */
        int32_t cap_len = 0;
        if (dl->current_shard_version >= 2) {
            cap_len = *(const int32_t *)(txt + stored_len);
        }
        const uint8_t *pixels = (const uint8_t *)(txt + stored_len + 1) + cap_len;

        /* Validate sample fits in buffer (accounting for optional caption) */
        long expected_size = 8 + (long)stored_len * 4 + px_bytes;
        if (dl->current_shard_version >= 2)
            expected_size += 4 + cap_len;
        if (sample_off + expected_size > dl->buffer_bytes)
            return -1;

        int full_len = stored_len + 1;
        text_lens[bs_actual] = full_len;
        if (label_ids) label_ids[bs_actual] = sample_hdr[0];

        int *in_row = (int *)tensor_data_ptr(input_ids) + (long)bs_actual * input_ids->strides[0];
        in_row[0] = TOKENIZER_BOS_ID;
        memcpy(in_row + 1, txt, (size_t)stored_len * sizeof(int));
        for (int t = full_len; t < T_max; t++) in_row[t] = IMAGENET_PAD_ID;

        int *tgt_row = (int *)tensor_data_ptr(target_ids) + (long)bs_actual * target_ids->strides[0];
        memcpy(tgt_row, txt, (size_t)stored_len * sizeof(int));
        tgt_row[stored_len] = IMAGENET_PAD_ID;
        for (int t = stored_len + 1; t < T_max; t++) tgt_row[t] = IMAGENET_PAD_ID;

        float *lm_row = (float *)tensor_data_ptr(loss_mask) + (long)bs_actual * loss_mask->strides[0];
        /* Decaying early-token weights: emphasize first bytes where vision
         * signal is strongest, reduce as text context accumulates. */
        for (int t = 0; t < stored_len; t++) {
            float w = 1.0f;
            if      (t == 0) w = 32.0f;
            else if (t == 1) w = 16.0f;
            else if (t == 2) w = 8.0f;
            else if (t == 3) w = 2.0f;
            lm_row[t] = w;
        }
        for (int t = stored_len; t < T_max; t++) lm_row[t] = 0.0f;

        float *img_row = tensor_data_ptr(img) + (long)bs_actual * img->strides[0];
        int flip = dl->shuffle ? (int)(xorshift64(&dl->rng_state) & 1) : 0;
        convert_pixels(img_row, pixels, H, W, C, dl->mean, dl->std, flip);
    }

    return bs_actual;
}
