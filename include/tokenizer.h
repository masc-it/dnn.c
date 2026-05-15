#ifndef DNN_TOKENIZER_H
#define DNN_TOKENIZER_H

#include "tensor.h"

/* ── Byte-Level Tokenizer ──
 *
 * Maps each byte value (0–255) directly to token ID = byte value.
 * Special tokens occupy IDs 257–260 (after the 256 byte values).
 *
 * Vocab size = 261.
 *
 * Encode prepends BOS and appends EOS.
 * Decode strips BOS, EOS, PAD. UNK maps to '?'.
 *
 * Optional special-token strings (e.g. "<|im_start|>") can be set
 * via bos_str / eos_str / pad_str.  When non-NULL, the encoder
 * scans the input text and replaces occurrences with the corresponding
 * special ID rather than encoding byte-by-byte.
 */

#define TOKENIZER_VOCAB_SIZE  261
#define TOKENIZER_BOS_ID      257
#define TOKENIZER_EOS_ID      258
#define TOKENIZER_PAD_ID      259
#define TOKENIZER_UNK_ID      260

/* ── Binary dataset format ──
 *
 * Header (64 bytes), then flat int32 token IDs:
 *
 *   [0x00] magic        = TOKENIZER_DATA_MAGIC
 *   [0x04] version      = TOKENIZER_DATA_VERSION
 *   [0x08] vocab_size
 *   [0x0C] num_sequences
 *   [0x10] seq_len       (includes BOS + EOS)
 *   [0x14] reserved[44]  (zero-filled, for forward compat)
 *   [0x40] data          num_sequences × seq_len × int32_t
 */

#define TOKENIZER_DATA_MAGIC       0x444E4E44  /* "DNND" */
#define TOKENIZER_DATA_VERSION     1
#define TOKENIZER_DATA_HEADER_SIZE 64  /* total header bytes */

typedef struct {
    int  bos_id;
    int  eos_id;
    int  pad_id;
    int  unk_id;
    const char *bos_str;   /* optional, e.g. "<|im_start|>" */
    const char *eos_str;   /* optional, e.g. "<|im_end|>"   */
    const char *pad_str;   /* optional, e.g. "<|pad|>"      */
    int  bos_len;          /* strlen(bos_str), 0 if NULL */
    int  eos_len;          /* strlen(eos_str), 0 if NULL */
    int  pad_len;          /* strlen(pad_str), 0 if NULL */
} tokenizer;

/* ── Lifecycle ── */

/* Create default tokenizer (BOS=257, EOS=258, PAD=259, UNK=260). */
tokenizer tokenizer_default(void);

/* Create tokenizer with chat-template special token strings:
 *   bos_str = "<|im_start|>"
 *   eos_str = "<|im_end|>"
 *   pad_str = "<|pad|>"
 * Special IDs same as default (BOS=257, EOS=258, PAD=259).
 */
tokenizer tokenizer_with_chat_template(void);

/* Create tokenizer from explicit special strings (NULL = skip). */
tokenizer tokenizer_with_specials(const char *bos_str,
                                   const char *eos_str,
                                   const char *pad_str);

/* ── Encode / Decode ── */

/* Encode text → int IDs array.
 *
 * Allocates from data pool.  Sets *len to number of IDs.
 * Prepends BOS, appends EOS.
 * If tok->{bos,eos,pad}_str are set, scans input for those
 * strings and emits the special ID instead of byte-by-byte.
 * Returns NULL on empty text (len=0).
 */
int *tokenizer_encode(struct mem_pool *data_pool, tokenizer *tok, const char *text, int *len);

/* Decode IDs → text string.
 *
 * Allocates with malloc — caller must free().
 * Strips BOS, EOS, PAD.  UNK maps to '?'.
 * Returns empty string (single NUL) on empty/blank input.
 */
char *tokenizer_decode(tokenizer *tok, const int *ids, int len);

/* ── Tensor pipeline ── */

/* Text → int tensor [1, N]  (N = encoded length).
 * Tensor allocated from data pool.
 */
tensor *tokenizer_text_to_tensor(struct mem_pool *data_pool, tokenizer *tok, const char *text);

/* Tensor → text string.
 * Reads ints from tensor data, decodes.
 * Allocates with malloc — caller must free().
 */
char *tokenizer_tensor_to_text(tokenizer *tok, const tensor *t);

#endif /* DNN_TOKENIZER_H */
