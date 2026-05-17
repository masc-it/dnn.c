#include "promessi.h"

/* ── Binary dataset loader ──
 *
 * Reads promessi.bin format:
 *   Header (64 bytes): magic(4) + version(4) + vocab_size(4)
 *                       + num_sequences(4) + seq_len(4) + reserved(44)
 *   Data: num_sequences × seq_len × int32_t token IDs
 */

lm_dataset load_dataset(const char *path) {
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
void shuffle_int(int *arr, int n) {
    for (int i = n - 1; i > 0; i--) {
        int j = dnn_rng_uniform_int(dnn_get_rng(), i + 1);
        int t = arr[i]; arr[i] = arr[j]; arr[j] = t;
    }
}
