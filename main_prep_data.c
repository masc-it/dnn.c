#include "dnn.h"
#include "tokenizer.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <errno.h>

/* ── Binary dataset format ──
 *
 * 64-byte header defined in tokenizer.h:
 *   magic, version, vocab_size, num_sequences, seq_len, reserved[44]
 * then num_sequences × seq_len × int32_t token IDs.
 */

static int read_file(const char *path, char **buf, size_t *len) {
    FILE *f = fopen(path, "rb");
    if (!f) {
        fprintf(stderr, "error: cannot open '%s': %s\n", path, strerror(errno));
        return -1;
    }

    if (fseek(f, 0, SEEK_END) != 0) {
        fprintf(stderr, "error: fseek failed on '%s': %s\n", path, strerror(errno));
        fclose(f);
        return -1;
    }

    long file_size = ftell(f);
    if (file_size < 0) {
        fprintf(stderr, "error: ftell failed on '%s': %s\n", path, strerror(errno));
        fclose(f);
        return -1;
    }

    rewind(f);

    *buf = (char *)malloc((size_t)file_size + 1);
    if (!*buf) {
        fprintf(stderr, "error: malloc(%ld) failed\n", file_size + 1);
        fclose(f);
        return -1;
    }

    size_t nread = fread(*buf, 1, (size_t)file_size, f);
    fclose(f);

    if ((long)nread != file_size) {
        fprintf(stderr, "error: read %zu/%ld bytes from '%s'\n",
                nread, file_size, path);
        free(*buf);
        *buf = NULL;
        return -1;
    }

    (*buf)[nread] = '\0';
    *len = nread;
    return 0;
}

static int ensure_data_dir(void) {
    struct stat st;
    if (stat("data", &st) == 0) {
        if (S_ISDIR(st.st_mode)) return 0;
        fprintf(stderr, "error: 'data' exists but is not a directory\n");
        return -1;
    }
#ifdef _WIN32
    if (mkdir("data") != 0) {
#else
    if (mkdir("data", 0755) != 0) {
#endif
        fprintf(stderr, "error: cannot create data/ dir: %s\n", strerror(errno));
        return -1;
    }
    return 0;
}

/* ── Usage ── */

static void print_usage(void) {
    fprintf(stderr,
        "Usage: main_prep_data --in-file <path> --chunk-size <N> --out-file <name>\n"
        "\n"
        "  --in-file    path to input .txt file\n"
        "  --chunk-size number of tokens per sequence (incl. BOS/EOS, min 3)\n"
        "  --out-file   output filename (saved as data/<name>.bin)\n"
        "\n"
        "Reads a .txt file, tokenizes with byte-level tokenizer (+ chat template),\n"
        "chunks token IDs into sequences of (chunk_size - 2) content tokens,\n"
        "wraps each in BOS/EOS, drops the last partial sequence, and saves\n"
        "as a binary .bin file.\n");
}

int main(int argc, char **argv) {
    const char *in_file  = NULL;
    const char *out_file = NULL;
    int chunk_size = 0;

    /* ── Parse args ── */
    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "--in-file") == 0 && i + 1 < argc) {
            in_file = argv[++i];
        } else if (strcmp(argv[i], "--chunk-size") == 0 && i + 1 < argc) {
            chunk_size = atoi(argv[++i]);
            if (chunk_size <= 0) {
                fprintf(stderr, "error: --chunk-size must be > 0, got %d\n", chunk_size);
                return 1;
            }
        } else if (strcmp(argv[i], "--out-file") == 0 && i + 1 < argc) {
            out_file = argv[++i];
        } else if (strcmp(argv[i], "--help") == 0 || strcmp(argv[i], "-h") == 0) {
            print_usage();
            return 0;
        } else {
            fprintf(stderr, "error: unknown arg '%s'\n", argv[i]);
            print_usage();
            return 1;
        }
    }

    if (!in_file || !out_file || chunk_size <= 0) {
        fprintf(stderr, "error: --in-file, --chunk-size, and --out-file are required\n");
        print_usage();
        return 1;
    }

    /* ── Ensure output dir ── */
    if (ensure_data_dir() != 0) return 1;

    /* ── Read input file ── */
    char *file_buf = NULL;
    size_t file_len = 0;
    if (read_file(in_file, &file_buf, &file_len) != 0) {
        return 1;
    }
    printf("Read %zu bytes from '%s'\n", file_len, in_file);

    /* ── Init pools (text may be large, size data pool accordingly) ── */
    size_t data_pool_size = file_len + 64 * 1024 * 1024;  /* file + token IDs + overhead */
    if (data_pool_size < 128 * 1024 * 1024)
        data_pool_size = 128 * 1024 * 1024;

    mem_pool params  = mem_pool_create(1 * 1024 * 1024);
    mem_pool scratch = mem_pool_create(16 * 1024 * 1024);
    mem_pool data    = mem_pool_create(data_pool_size);
    mem_pool_set_defaults(&params, &scratch, &data);

    /* ── Tokenize ── */
    tokenizer tok = tokenizer_with_chat_template();

    int total_ids_len;
    int *all_ids = tokenizer_encode(&tok, file_buf, &total_ids_len);
    if (!all_ids || total_ids_len <= 0) {
        fprintf(stderr, "error: tokenizer returned empty\n");
        free(file_buf);
        mem_pool_destroy(&params);
        mem_pool_destroy(&scratch);
        mem_pool_destroy(&data);
        return 1;
    }
    printf("Tokenized: %d IDs (including outer BOS/EOS)\n", total_ids_len);

    free(file_buf);

    /* ── Strip outer BOS/EOS added by tokenizer_encode ── */
    /* all_ids[0] = BOS, all_ids[total_ids_len-1] = EOS */
    int content_len = total_ids_len - 2;
    int *content_start = all_ids + 1;

    /* ── Chunk ── */
    int content_per_chunk = chunk_size - 2;  /* reserve 1 BOS + 1 EOS */
    int seq_len          = chunk_size;        /* total including BOS/EOS */
    if (chunk_size < 3) {
        fprintf(stderr, "error: --chunk-size must be >= 3 (need room for BOS + 1 token + EOS)\n");
        mem_pool_destroy(&params);
        mem_pool_destroy(&scratch);
        mem_pool_destroy(&data);
        return 1;
    }
    int num_chunks = content_len / content_per_chunk;
    int num_sequences = num_chunks;

    if (num_chunks == 0) {
        fprintf(stderr, "error: content too short (%d tokens) for chunk_size=%d\n",
                content_len, chunk_size);
        mem_pool_destroy(&params);
        mem_pool_destroy(&scratch);
        mem_pool_destroy(&data);
        return 1;
    }

    int total_dropped = content_len - num_chunks * content_per_chunk;
    printf("Chunks: %d sequences of %d tokens (%d content + BOS/EOS)",
           num_chunks, seq_len, content_per_chunk);
    if (total_dropped > 0)
        printf(" (dropped %d trailing tokens)", total_dropped);
    printf("\n");

    /* ── Build output path: data/<out_file>.bin ── */
    size_t out_path_len = strlen("data/") + strlen(out_file) + strlen(".bin") + 1;
    char *out_path = (char *)malloc(out_path_len);
    if (!out_path) {
        fprintf(stderr, "error: malloc failed for output path\n");
        mem_pool_destroy(&params);
        mem_pool_destroy(&scratch);
        mem_pool_destroy(&data);
        return 1;
    }
    snprintf(out_path, out_path_len, "data/%s.bin", out_file);

    /* ── Write binary ── */
    FILE *fout = fopen(out_path, "wb");
    if (!fout) {
        fprintf(stderr, "error: cannot create '%s': %s\n", out_path, strerror(errno));
        free(out_path);
        mem_pool_destroy(&params);
        mem_pool_destroy(&scratch);
        mem_pool_destroy(&data);
        return 1;
    }

    /* ── Write 64-byte header ── */
    struct {
        int32_t magic;
        int32_t version;
        int32_t vocab_size;
        int32_t num_sequences;
        int32_t seq_len;
        char    reserved[TOKENIZER_DATA_HEADER_SIZE - 5 * sizeof(int32_t)];
    } hdr;

    hdr.magic         = (int32_t)TOKENIZER_DATA_MAGIC;
    hdr.version       = (int32_t)TOKENIZER_DATA_VERSION;
    hdr.vocab_size    = (int32_t)TOKENIZER_VOCAB_SIZE;
    hdr.num_sequences = (int32_t)num_sequences;
    hdr.seq_len       = (int32_t)seq_len;
    memset(hdr.reserved, 0, sizeof(hdr.reserved));

    if (fwrite(&hdr, TOKENIZER_DATA_HEADER_SIZE, 1, fout) != 1) {
        fprintf(stderr, "error: writing header to '%s'\n", out_path);
        fclose(fout);
        free(out_path);
        mem_pool_destroy(&params);
        mem_pool_destroy(&scratch);
        mem_pool_destroy(&data);
        return 1;
    }

    /* ── Write sequences: each = [BOS, chunk_size content tokens, EOS] ── */
    int32_t bos = (int32_t)TOKENIZER_BOS_ID;
    int32_t eos = (int32_t)TOKENIZER_EOS_ID;

    for (int s = 0; s < num_chunks; s++) {
        const int *chunk_start = content_start + s * content_per_chunk;

        /* write BOS */
        if (fwrite(&bos, sizeof(int32_t), 1, fout) != 1) goto write_err;

        /* write content_per_chunk content tokens */
        if (fwrite(chunk_start, sizeof(int32_t), (size_t)content_per_chunk, fout) != (size_t)content_per_chunk)
            goto write_err;

        /* write EOS */
        if (fwrite(&eos, sizeof(int32_t), 1, fout) != 1) goto write_err;
    }

    fclose(fout);
    printf("Wrote %s  (%d sequences × %d tokens = %zu bytes, header=%d)\n",
           out_path, num_sequences, seq_len,
           (size_t)TOKENIZER_DATA_HEADER_SIZE + (size_t)num_sequences * seq_len * sizeof(int32_t),
           TOKENIZER_DATA_HEADER_SIZE);

    free(out_path);
    mem_pool_destroy(&params);
    mem_pool_destroy(&scratch);
    mem_pool_destroy(&data);
    return 0;

write_err:
    fprintf(stderr, "error: writing sequence data to '%s'\n", out_path);
    fclose(fout);
    free(out_path);
    mem_pool_destroy(&params);
    mem_pool_destroy(&scratch);
    mem_pool_destroy(&data);
    return 1;
}
