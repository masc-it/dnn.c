#include "mnist.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <assert.h>
#include <zlib.h>

/* ================================================================
 *  Internal helpers
 * ================================================================ */

/* read big-endian 32-bit integer */
static int read_be32(const unsigned char *p) {
    return ((int)p[0] << 24) | ((int)p[1] << 16) | ((int)p[2] << 8) | p[3];
}

/* parse IDX header, fill dims[] with ndim values.  returns 0 on success. */
static int read_idx_header(gzFile f, int *magic, int *dims) {
    unsigned char hdr[16];
    if (gzread(f, hdr, 4) < 4) return -1;
    *magic = read_be32(hdr);
    int ndim = *magic & 0xFF;
    if (gzread(f, hdr, ndim * 4) < ndim * 4) return -1;
    for (int i = 0; i < ndim; i++) dims[i] = read_be32(hdr + i * 4);
    return 0;
}

/* ================================================================
 *  Data loading
 * ================================================================ */

int mnist_download(void) {
    /* ensure data directory exists */
    int r = system("mkdir -p " MNIST_DATA_DIR);
    if (r != 0) {
        fprintf(stderr, "mnist_download: failed to create %s\n", MNIST_DATA_DIR);
        return -1;
    }

    static const char *urls[] = {
        "https://github.com/fgnt/mnist/raw/refs/heads/master/"
        "train-images-idx3-ubyte.gz",
        "https://github.com/fgnt/mnist/raw/refs/heads/master/"
        "train-labels-idx1-ubyte.gz",
        "https://github.com/fgnt/mnist/raw/refs/heads/master/"
        "t10k-images-idx3-ubyte.gz",
        "https://github.com/fgnt/mnist/raw/refs/heads/master/"
        "t10k-labels-idx1-ubyte.gz",
    };
    static const char *names[] = {
        "train-images-idx3-ubyte",
        "train-labels-idx1-ubyte",
        "t10k-images-idx3-ubyte",
        "t10k-labels-idx1-ubyte",
    };
    int n = 4;

    for (int i = 0; i < n; i++) {
        /* skip if already downloaded */
        char path[512];
        r = snprintf(path, sizeof(path), MNIST_DATA_DIR "/%s", names[i]);
        assert(r > 0 && r < (int)sizeof(path));

        FILE *f = fopen(path, "rb");
        if (f) { fclose(f); continue; }

        char cmd[1024];
        r = snprintf(cmd, sizeof(cmd),
                     "curl -fSL '%s' 2>/dev/null | gunzip > '%s'",
                     urls[i], path);
        assert(r > 0 && r < (int)sizeof(cmd));

        int ret = system(cmd);
        if (ret != 0) {
            fprintf(stderr, "mnist_download: failed (%d) for %s\n",
                    ret, urls[i]);
            return -1;
        }
    }
    return 0;
}

tensor *mnist_load_images(struct mem_pool *data, const char *path) {
    gzFile f = gzopen(path, "rb");
    if (!f) { fprintf(stderr, "mnist_load_images: can't open %s\n", path); return NULL; }

    int magic, dims[4];
    if (read_idx_header(f, &magic, dims) < 0) { gzclose(f); return NULL; }

    int N = dims[0], rows = dims[1], cols = dims[2];
    int n_pixels = N * rows * cols;
    int img_size = rows * cols;  /* 784 */

    /* read raw unsigned-byte pixels */
    unsigned char *raw = malloc(n_pixels);
    if (gzread(f, raw, n_pixels) < n_pixels) {
        free(raw); gzclose(f); return NULL;
    }
    gzclose(f);

    /* create float tensor [N, 784] in data pool, values in [0, 1] */
    tensor *t = tensor_zeros_data(data, 2, (int[]){N, img_size});
    float  *dp = (float *)t->data;
    for (int i = 0; i < n_pixels; i++)
        dp[i] = raw[i] / 255.0f;

    free(raw);
    return t;
}

tensor *mnist_load_labels(struct mem_pool *data, const char *path) {
    gzFile f = gzopen(path, "rb");
    if (!f) { fprintf(stderr, "mnist_load_labels: can't open %s\n", path); return NULL; }

    int magic, dims[4];
    if (read_idx_header(f, &magic, dims) < 0) { gzclose(f); return NULL; }

    int N = dims[0];

    unsigned char *raw = malloc(N);
    if (gzread(f, raw, N) < N) { free(raw); gzclose(f); return NULL; }
    gzclose(f);

    /* create int tensor [N] in data pool – store ints in the float data region */
    tensor *t = tensor_zeros_data(data, 1, (int[]){N});
    int    *lp = (int *)t->data;
    for (int i = 0; i < N; i++) lp[i] = raw[i];

    free(raw);
    return t;
}
