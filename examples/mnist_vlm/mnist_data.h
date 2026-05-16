/* ══════════════════════════════════════════════════════════════════
 *  Shared MNIST data loading (IDX format)
 *
 *  Included by mnist_vlm.c and mnist_vlm_preds.c.
 *  Uses zlib (gzopen/gzread) for gzipped IDX files.
 * ══════════════════════════════════════════════════════════════════ */

#ifndef MNIST_DATA_H
#define MNIST_DATA_H

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <zlib.h>

#define DATA_DIR "data/mnist"

static int mnist_read_be32(const unsigned char *p) {
    return ((int)p[0] << 24) | ((int)p[1] << 16) | ((int)p[2] << 8) | p[3];
}

/* Load MNIST image IDX file → float array [N, rows, cols], pixels in [0,1]. */
static float *mnist_load_images(const char *path, int *n_out) {
    gzFile f = gzopen(path, "rb");
    if (!f) { fprintf(stderr, "can't open %s\n", path); return NULL; }
    unsigned char hdr[16];
    if (gzread(f, hdr, 16) < 16) { gzclose(f); return NULL; }
    int N = mnist_read_be32(hdr + 4);
    int rows = mnist_read_be32(hdr + 8);
    int cols = mnist_read_be32(hdr + 12);
    *n_out = N;
    int total = N * rows * cols;
    unsigned char *raw = malloc(total);
    if (!raw) { gzclose(f); return NULL; }
    if (gzread(f, raw, total) < total) { free(raw); gzclose(f); return NULL; }
    gzclose(f);
    float *data = malloc((size_t)total * sizeof(float));
    if (!data) { free(raw); return NULL; }
    for (int i = 0; i < total; i++) data[i] = raw[i] / 255.0f;
    free(raw);
    return data;
}

/* Load MNIST label IDX file → int array [N]. */
static int *mnist_load_labels(const char *path, int *n_out) {
    gzFile f = gzopen(path, "rb");
    if (!f) { fprintf(stderr, "can't open %s\n", path); return NULL; }
    unsigned char hdr[8];
    if (gzread(f, hdr, 8) < 8) { gzclose(f); return NULL; }
    int N = mnist_read_be32(hdr + 4);
    *n_out = N;
    unsigned char *raw = malloc(N);
    if (!raw) { gzclose(f); return NULL; }
    if (gzread(f, raw, N) < N) { free(raw); gzclose(f); return NULL; }
    gzclose(f);
    int *data = malloc((size_t)N * sizeof(int));
    if (!data) { free(raw); return NULL; }
    for (int i = 0; i < N; i++) data[i] = raw[i];
    free(raw);
    return data;
}

#endif /* MNIST_DATA_H */
