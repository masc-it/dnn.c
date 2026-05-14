#include "tokenizer.h"
#include "pool.h"
#include <string.h>
#include <stdlib.h>
#include <assert.h>

/* ── Constructors ── */

tokenizer tokenizer_default(void) {
    tokenizer tok;
    tok.bos_id   = TOKENIZER_BOS_ID;
    tok.eos_id   = TOKENIZER_EOS_ID;
    tok.pad_id   = TOKENIZER_PAD_ID;
    tok.unk_id   = TOKENIZER_UNK_ID;
    tok.bos_str  = NULL;
    tok.eos_str  = NULL;
    tok.pad_str  = NULL;
    tok.bos_len  = 0;
    tok.eos_len  = 0;
    tok.pad_len  = 0;
    return tok;
}

tokenizer tokenizer_with_chat_template(void) {
    return tokenizer_with_specials("<|im_start|>", "<|im_end|>", "<|pad|>");
}

tokenizer tokenizer_with_specials(const char *bos_str,
                                   const char *eos_str,
                                   const char *pad_str) {
    tokenizer tok = tokenizer_default();
    tok.bos_str = bos_str;
    tok.eos_str = eos_str;
    tok.pad_str = pad_str;
    tok.bos_len = bos_str ? (int)strlen(bos_str) : 0;
    tok.eos_len = eos_str ? (int)strlen(eos_str) : 0;
    tok.pad_len = pad_str ? (int)strlen(pad_str) : 0;
    return tok;
}

/* ── Internal: scan for special token strings ──
 *
 * At position pos in text, check if any special string matches.
 * Returns the special ID to emit, or -1 if no match.
 * Sets *match_len to the length of the matched string (must advance by this).
 */

struct _spec_entry {
    int       id;
    const char *str;
    int       len;
};

static int _match_special(tokenizer *tok, const char *text, int pos,
                           int text_len, int *match_len) {
    struct _spec_entry entries[3];
    int n_spec = 0;

    if (tok->bos_str && tok->bos_len > 0) {
        entries[n_spec].id  = tok->bos_id;
        entries[n_spec].str = tok->bos_str;
        entries[n_spec].len = tok->bos_len;
        n_spec++;
    }
    if (tok->eos_str && tok->eos_len > 0) {
        entries[n_spec].id  = tok->eos_id;
        entries[n_spec].str = tok->eos_str;
        entries[n_spec].len = tok->eos_len;
        n_spec++;
    }
    if (tok->pad_str && tok->pad_len > 0) {
        entries[n_spec].id  = tok->pad_id;
        entries[n_spec].str = tok->pad_str;
        entries[n_spec].len = tok->pad_len;
        n_spec++;
    }

    for (int s = 0; s < n_spec; s++) {
        if (pos + entries[s].len > text_len)
            continue;
        if (memcmp(text + pos, entries[s].str, (size_t)entries[s].len) == 0) {
            *match_len = entries[s].len;
            return entries[s].id;
        }
    }
    return -1;
}

/* ── Encode ── */

int *tokenizer_encode(tokenizer *tok, const char *text, int *len) {
    if (!text || text[0] == '\0') {
        *len = 0;
        return NULL;
    }

    int text_len = (int)strlen(text);

    /* Max possible IDs: BOS + every byte as its own ID + EOS = text_len + 2.
     * Actual count ≤ this (specials compress multi-byte strings to 1 ID). */
    int max_ids = text_len + 2;

    int *ids = (int *)mem_data_alloc((size_t)max_ids * sizeof(int), NULL);
    int out = 0;

    ids[out++] = tok->bos_id;

    int i = 0;
    while (i < text_len) {
        int match_len;
        int spec_id = _match_special(tok, text, i, text_len, &match_len);
        if (spec_id >= 0) {
            ids[out++] = spec_id;
            i += match_len;
        } else {
            ids[out++] = (unsigned char)text[i];
            i++;
        }
    }

    ids[out++] = tok->eos_id;

    *len = out;
    return ids;
}

/* ── Decode ── */

char *tokenizer_decode(tokenizer *tok, const int *ids, int len) {
    if (!ids || len <= 0) {
        char *empty = (char *)malloc(1);
        if (empty) empty[0] = '\0';
        return empty;
    }

    /* Output buffer: at most len bytes (worst case: all valid bytes) + NUL */
    char *text = (char *)malloc((size_t)len + 1);
    if (!text) return NULL;

    int out = 0;
    for (int i = 0; i < len; i++) {
        int id = ids[i];
        if (id == tok->bos_id || id == tok->eos_id || id == tok->pad_id) {
            continue;  /* strip special tokens */
        }
        if (id >= 0 && id < 256) {
            text[out++] = (char)(unsigned char)id;
        } else {
            text[out++] = '?';  /* UNK or any unknown ID */
        }
    }
    text[out] = '\0';
    return text;
}

/* ── Tensor pipeline ── */

tensor *tokenizer_text_to_tensor(tokenizer *tok, const char *text) {
    int len;
    int *ids = tokenizer_encode(tok, text, &len);
    if (!ids || len <= 0) {
        return NULL;
    }

    /* tensor_zeros_data allocates float-sized elements; sizeof(float)=4
     * matches sizeof(int) so ints fit in the same buffer. */
    int shape[] = {1, len};
    tensor *t = tensor_zeros_data(2, shape);
    memcpy(t->data, ids, (size_t)len * sizeof(int));
    return t;
}

char *tokenizer_tensor_to_text(tokenizer *tok, const tensor *t) {
    if (!t || tensor_numel(t) == 0) {
        char *empty = (char *)malloc(1);
        if (empty) empty[0] = '\0';
        return empty;
    }
    int len = tensor_numel(t);
    const int *ids = (const int *)t->data;
    return tokenizer_decode(tok, ids, len);
}
