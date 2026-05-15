#include "tokenizer.h"
#include "context.h"
#include "pool.h"
#include "context.h"
#include <stdio.h>
#include <assert.h>
#include <string.h>
#include <stdlib.h>

static dnn_ctx ctx;

/* ── helpers ── */

/* ── Tests ── */

static void test_encode_decode_roundtrip(void) {
    printf("  test_encode_decode_roundtrip... ");
    tokenizer tok = tokenizer_default();

    const char *original = "hello";
    int len;
    int *ids = tokenizer_encode(ctx.data, &tok, original, &len);
    assert(len == 7);  /* BOS + 5 + EOS */
    assert(ids[0] == TOKENIZER_BOS_ID);
    assert(ids[1] == 'h');
    assert(ids[2] == 'e');
    assert(ids[3] == 'l');
    assert(ids[4] == 'l');
    assert(ids[5] == 'o');
    assert(ids[6] == TOKENIZER_EOS_ID);

    char *decoded = tokenizer_decode(&tok, ids, len);
    assert(strcmp(decoded, original) == 0);
    free(decoded);
    printf("OK\n");
}

static void test_encode_decode_empty(void) {
    printf("  test_encode_decode_empty... ");
    tokenizer tok = tokenizer_default();

    /* empty string encode */
    int len;
    int *ids = tokenizer_encode(ctx.data, &tok, "", &len);
    assert(ids == NULL);
    assert(len == 0);

    /* empty decode */
    char *text = tokenizer_decode(&tok, NULL, 0);
    assert(text != NULL);
    assert(text[0] == '\0');
    free(text);
    printf("OK\n");
}

static void test_encode_special_tokens_stripped(void) {
    printf("  test_encode_special_tokens_stripped... ");
    tokenizer tok = tokenizer_default();

    /* Decode with only special tokens → empty */
    int ids[] = {TOKENIZER_BOS_ID, TOKENIZER_EOS_ID};
    char *text = tokenizer_decode(&tok, ids, 2);
    assert(strcmp(text, "") == 0);
    free(text);

    /* Decode with PAD stripped */
    int ids2[] = {'a', TOKENIZER_PAD_ID, 'b'};
    text = tokenizer_decode(&tok, ids2, 3);
    assert(strcmp(text, "ab") == 0);
    free(text);

    /* Decode with UNK mapped to '?' */
    int ids3[] = {'a', 999, 'b'};
    text = tokenizer_decode(&tok, ids3, 3);
    assert(strcmp(text, "a?b") == 0);
    free(text);
    printf("OK\n");
}

static void test_encode_utf8_bytes(void) {
    printf("  test_encode_utf8_bytes... ");
    tokenizer tok = tokenizer_default();

    /* UTF-8 encoded "caffè" — è is 2 bytes (0xC3 0xA8) */
    const char *utf8 = "caff\xC3\xA8";
    int len;
    int *ids = tokenizer_encode(ctx.data, &tok, utf8, &len);
    assert(len == 8);  /* BOS + 6 bytes + EOS */

    /* bytes: c(0x63) a(0x61) f(0x66) f(0x66) 0xC3 0xA8 */
    assert(ids[1] == 0x63);
    assert(ids[2] == 0x61);
    assert(ids[3] == 0x66);
    assert(ids[4] == 0x66);
    assert(ids[5] == 0xC3);
    assert(ids[6] == 0xA8);

    /* roundtrip preserves raw bytes */
    char *decoded = tokenizer_decode(&tok, ids, len);
    assert(strcmp(decoded, utf8) == 0);
    free(decoded);
    printf("OK\n");
}

static void test_encode_all_byte_values(void) {
    printf("  test_encode_all_byte_values... ");
    tokenizer tok = tokenizer_default();

    /* Build string containing byte values 1..255 followed by NUL.
     * Skip 0 so strlen works (no embedded NUL). */
    unsigned char buf[256];
    for (int i = 0; i < 255; i++) buf[i] = (unsigned char)(i + 1);
    buf[255] = '\0';
    const char *text = (const char *)buf;

    int len;
    int *ids = tokenizer_encode(ctx.data, &tok, text, &len);
    assert(len == 257);  /* BOS + 255 bytes + EOS */

    for (int i = 0; i < 255; i++) {
        assert(ids[i + 1] == (int)(i + 1));
    }

    /* decode and compare raw bytes */
    char *decoded = tokenizer_decode(&tok, ids, len);
    assert((int)strlen(decoded) == 255);
    for (int i = 0; i < 255; i++) {
        assert((unsigned char)decoded[i] == (unsigned char)(i + 1));
    }
    free(decoded);
    printf("OK\n");
}

static void test_text_to_tensor_pipeline(void) {
    printf("  test_text_to_tensor_pipeline... ");
    tokenizer tok = tokenizer_default();

    const char *original = "hello world";
    tensor *t = tokenizer_text_to_tensor(ctx.data, &tok, original);
    assert(t != NULL);
    assert(tensor_ndim(t) == 2);
    assert(tensor_shape(t, 0) == 1);
    /* encoded = BOS(1) + 11 bytes + EOS(1) = 13 */
    int expected_len = (int)strlen(original) + 2;
    assert(tensor_shape(t, 1) == expected_len);

    /* verify data via int access */
    const int *ids = (const int *)t->data;
    assert(ids[0] == TOKENIZER_BOS_ID);
    for (size_t i = 0; i < strlen(original); i++) {
        assert(ids[i + 1] == (unsigned char)original[i]);
    }
    assert(ids[expected_len - 1] == TOKENIZER_EOS_ID);

    /* tensor -> text roundtrip */
    char *decoded = tokenizer_tensor_to_text(&tok, t);
    assert(strcmp(decoded, original) == 0);
    free(decoded);
    printf("OK\n");
}

static void test_chat_template_encode(void) {
    printf("  test_chat_template_encode... ");
    tokenizer tok = tokenizer_with_chat_template();
    assert(tok.bos_id == TOKENIZER_BOS_ID);
    assert(tok.eos_id == TOKENIZER_EOS_ID);
    assert(tok.pad_id == TOKENIZER_PAD_ID);
    assert(tok.bos_str != NULL && strcmp(tok.bos_str, "<|im_start|>") == 0);
    assert(tok.eos_str != NULL && strcmp(tok.eos_str, "<|im_end|>") == 0);
    assert(tok.pad_str != NULL && strcmp(tok.pad_str, "<|pad|>") == 0);

    /* encode: "hello" without special strings — same as default */
    int len;
    int *ids = tokenizer_encode(ctx.data, &tok, "hello", &len);
    assert(len == 7);
    assert(ids[0] == TOKENIZER_BOS_ID);
    assert(ids[1] == 'h');
    assert(ids[6] == TOKENIZER_EOS_ID);
    printf("OK\n");
}

static void test_chat_template_maps_special_strings(void) {
    printf("  test_chat_template_maps_special_strings... ");
    tokenizer tok = tokenizer_with_chat_template();

    /* "<|im_start|>" is 12 bytes → 1 token (BOS)
     * "hello" is 5 bytes → 5 tokens
     * "<|im_end|>" is 10 bytes → 1 token (EOS)
     * content tokens: 1 + 5 + 1 = 7
     * plus auto BOS + auto EOS = 7 + 2 = 9
     */
    int len;
    int *ids = tokenizer_encode(ctx.data, &tok, "<|im_start|>hello<|im_end|>", &len);
    assert(len == 9);
    assert(ids[0] == TOKENIZER_BOS_ID);    /* auto BOS */
    assert(ids[1] == TOKENIZER_BOS_ID);    /* <|im_start|> → BOS */
    assert(ids[2] == 'h');
    assert(ids[3] == 'e');
    assert(ids[4] == 'l');
    assert(ids[5] == 'l');
    assert(ids[6] == 'o');
    assert(ids[7] == TOKENIZER_EOS_ID);    /* <|im_end|> → EOS */
    assert(ids[8] == TOKENIZER_EOS_ID);    /* auto EOS */

    /* decode strips both auto and embedded specials */
    char *text = tokenizer_decode(&tok, ids, len);
    assert(strcmp(text, "hello") == 0);
    free(text);
    printf("OK\n");
}

static void test_chat_template_overlap_non_special(void) {
    printf("  test_chat_template_overlap_non_special... ");
    tokenizer tok = tokenizer_with_chat_template();

    /* text that looks like a prefix of special but isn't complete */
    int len;
    int *ids = tokenizer_encode(ctx.data, &tok, "<|im_not_a_token|>", &len);
    /* all bytes encoded individually (no match) */
    int expected = (int)strlen("<|im_not_a_token|>") + 2;
    assert(len == expected);
    assert(ids[1] == '<');
    assert(ids[2] == '|');

    /* roundtrip */
    char *text = tokenizer_decode(&tok, ids, len);
    assert(strcmp(text, "<|im_not_a_token|>") == 0);
    free(text);
    printf("OK\n");
}

static void test_chat_template_pad_string(void) {
    printf("  test_chat_template_pad_string... ");
    tokenizer tok = tokenizer_with_chat_template();

    /* "a<|pad|>b" should become [BOS, a, PAD, b, EOS] */
    int len;
    int *ids = tokenizer_encode(ctx.data, &tok, "a<|pad|>b", &len);
    assert(len == 5);
    assert(ids[0] == TOKENIZER_BOS_ID);
    assert(ids[1] == 'a');
    assert(ids[2] == TOKENIZER_PAD_ID);
    assert(ids[3] == 'b');
    assert(ids[4] == TOKENIZER_EOS_ID);

    /* decode strips PAD */
    char *text = tokenizer_decode(&tok, ids, len);
    assert(strcmp(text, "ab") == 0);
    free(text);
    printf("OK\n");
}

static void test_chat_template_multiple_specials(void) {
    printf("  test_chat_template_multiple_specials... ");
    tokenizer tok = tokenizer_with_chat_template();

    /* multiple interleaved specials */
    const char *input = "<|im_start|>user<|im_end|><|im_start|>assistant<|im_end|>";
    int len;
    int *ids = tokenizer_encode(ctx.data, &tok, input, &len);

    /* expected: auto BOS + [BOS,user,EOS,BOS,assistant,EOS] + auto EOS */
    /* content tokens: BOS + user(4) + EOS + BOS + assistant(9) + EOS = 17 */
    /* total: 17 + 2 = 19 */
    assert(len == 19);
    assert(ids[0] == TOKENIZER_BOS_ID);   /* auto BOS */
    assert(ids[1] == TOKENIZER_BOS_ID);   /* <|im_start|> */
    assert(ids[2] == 'u');
    assert(ids[3] == 's');
    assert(ids[4] == 'e');
    assert(ids[5] == 'r');
    assert(ids[6] == TOKENIZER_EOS_ID);   /* <|im_end|> */
    assert(ids[7] == TOKENIZER_BOS_ID);   /* <|im_start|> */
    assert(ids[8] == 'a');
    assert(ids[9] == 's');
    assert(ids[10] == 's');
    assert(ids[11] == 'i');
    assert(ids[12] == 's');
    assert(ids[13] == 't');
    assert(ids[14] == 'a');
    assert(ids[15] == 'n');
    assert(ids[16] == 't');
    assert(ids[17] == TOKENIZER_EOS_ID);  /* <|im_end|> */
    assert(ids[18] == TOKENIZER_EOS_ID);  /* auto EOS */

    char *text = tokenizer_decode(&tok, ids, len);
    assert(strcmp(text, "userassistant") == 0);
    free(text);
    printf("OK\n");
}

static void test_chat_template_only_specials(void) {
    printf("  test_chat_template_only_specials... ");
    tokenizer tok = tokenizer_with_chat_template();

    /* text consisting entirely of special strings */
    int len;
    int *ids = tokenizer_encode(ctx.data, &tok, "<|im_start|><|im_end|>", &len);
    /* auto BOS + BOS + EOS + auto EOS = 4 */
    assert(len == 4);
    assert(ids[0] == TOKENIZER_BOS_ID);
    assert(ids[1] == TOKENIZER_BOS_ID);
    assert(ids[2] == TOKENIZER_EOS_ID);
    assert(ids[3] == TOKENIZER_EOS_ID);

    char *text = tokenizer_decode(&tok, ids, len);
    assert(strcmp(text, "") == 0);
    free(text);
    printf("OK\n");
}

static void test_chat_template_large_text_roundtrip(void) {
    printf("  test_chat_template_large_text_roundtrip... ");
    tokenizer tok = tokenizer_with_chat_template();

    /* Read first 5000 bytes from promessi_sposi.txt */
    FILE *f = fopen("docs/promessi_sposi.txt", "rb");
    assert(f != NULL);

    unsigned char raw[5001];
    size_t nread = fread(raw, 1, 5000, f);
    fclose(f);
    assert(nread > 0);
    raw[nread] = '\0';

    const char *text = (const char *)raw;
    int len;
    int *ids = tokenizer_encode(ctx.data, &tok, text, &len);
    assert(len == (int)nread + 2);  /* no special strings in Italian text */

    /* all bytes preserved */
    for (size_t i = 0; i < nread; i++) {
        assert(ids[i + 1] == (int)(unsigned char)text[i]);
    }

    char *decoded = tokenizer_decode(&tok, ids, len);
    assert(memcmp(decoded, text, nread) == 0);
    free(decoded);
    printf("OK (%zu bytes)\n", nread);
}

static void test_with_specials_custom(void) {
    printf("  test_with_specials_custom... ");
    tokenizer tok = tokenizer_with_specials("[BOS]", "[EOS]", "[PAD]");
    assert(tok.bos_id == TOKENIZER_BOS_ID);
    assert(strcmp(tok.bos_str, "[BOS]") == 0);
    assert(tok.bos_len == 5);

    int len;
    int *ids = tokenizer_encode(ctx.data, &tok, "a[PAD]b", &len);
    assert(len == 5);
    assert(ids[1] == 'a');
    assert(ids[2] == TOKENIZER_PAD_ID);
    assert(ids[3] == 'b');

    char *text = tokenizer_decode(&tok, ids, len);
    assert(strcmp(text, "ab") == 0);
    free(text);
    printf("OK\n");
}

static void test_custom_tokenizer(void) {
    printf("  test_custom_tokenizer... ");
    tokenizer tok;
    tok.bos_id = 100;
    tok.eos_id = 200;
    tok.pad_id = 0;
    tok.unk_id = 255;

    int len;
    int *ids = tokenizer_encode(ctx.data, &tok, "abc", &len);
    assert(len == 5);
    assert(ids[0] == 100);  /* custom BOS */
    assert(ids[1] == 'a');
    assert(ids[2] == 'b');
    assert(ids[3] == 'c');
    assert(ids[4] == 200);  /* custom EOS */

    /* decode strips custom specials */
    char *text = tokenizer_decode(&tok, ids, len);
    assert(strcmp(text, "abc") == 0);
    free(text);

    /* custom PAD (0) stripped */
    int ids2[] = {'a', 0, 'b'};
    text = tokenizer_decode(&tok, ids2, 3);
    assert(strcmp(text, "ab") == 0);
    free(text);
    printf("OK\n");
}

static void test_large_text_roundtrip(void) {
    printf("  test_large_text_roundtrip... ");
    tokenizer tok = tokenizer_default();

    /* Read first 5000 bytes from promessi_sposi.txt */
    FILE *f = fopen("docs/promessi_sposi.txt", "rb");
    assert(f != NULL && "cannot open promessi_sposi.txt");

    /* Read first 5000 bytes */
    unsigned char raw[5001];  /* +1 for NUL */
    size_t nread = fread(raw, 1, 5000, f);
    fclose(f);
    assert(nread > 0);
    raw[nread] = '\0';  /* ensure NUL termination */

    const char *text = (const char *)raw;

    int len;
    int *ids = tokenizer_encode(ctx.data, &tok, text, &len);
    assert(len == (int)nread + 2);  /* BOS + nread bytes + EOS */

    /* Check IDs directly */
    assert(ids[0] == TOKENIZER_BOS_ID);
    for (size_t i = 0; i < nread; i++) {
        if (ids[i + 1] != (int)(unsigned char)text[i]) {
            printf("    ID mismatch at byte %zu: got %d expected %d\n",
                   i, ids[i + 1], (int)(unsigned char)text[i]);
            assert(0);
        }
    }
    assert(ids[nread + 1] == TOKENIZER_EOS_ID);

    /* Decode back */
    char *decoded = tokenizer_decode(&tok, ids, len);
    assert((int)strlen(decoded) == (int)nread);

    /* Compare raw bytes */
    int match = (memcmp(decoded, text, nread) == 0);
    if (!match) {
        for (size_t i = 0; i < nread; i++) {
            if ((unsigned char)decoded[i] != raw[i]) {
                printf("    mismatch at byte %zu: got 0x%02x expected 0x%02x\n",
                       i, (unsigned char)decoded[i], raw[i]);
                break;
            }
        }
    }
    assert(match);
    free(decoded);
    printf("OK (%zu bytes)\n", nread);
}

static void test_tensor_to_text_empty_tensor(void) {
    printf("  test_tensor_to_text_empty_tensor... ");
    tokenizer tok = tokenizer_default();

    /* NULL tensor → empty string */
    char *text = tokenizer_tensor_to_text(&tok, NULL);
    assert(text != NULL);
    assert(text[0] == '\0');
    free(text);

    /* Tensor with 0 elements → empty string */
    int shape[] = {1, 0};
    tensor *t = tensor_zeros_data(ctx.data, 2, shape);
    text = tokenizer_tensor_to_text(&tok, t);
    assert(text != NULL);
    assert(text[0] == '\0');
    free(text);
    printf("OK\n");
}

static void test_text_to_tensor_null_empty(void) {
    printf("  test_text_to_tensor_null_empty... ");
    tokenizer tok = tokenizer_default();

    /* empty string → NULL */
    tensor *t = tokenizer_text_to_tensor(ctx.data, &tok, "");
    assert(t == NULL);

    printf("OK\n");
}

/* ── Main ── */

int main(void) {
    printf("test_tokenizer:\n");

    /* pools for tensor ops (text_to_tensor) */

    dnn_ctx_init(&ctx, 2 * 1024 * 1024, 2 * 1024 * 1024, 10 * 1024 * 1024);

    test_encode_decode_roundtrip();
    test_encode_decode_empty();
    test_encode_special_tokens_stripped();
    test_encode_utf8_bytes();
    test_encode_all_byte_values();
    test_text_to_tensor_pipeline();
    test_chat_template_encode();
    test_chat_template_maps_special_strings();
    test_chat_template_overlap_non_special();
    test_chat_template_pad_string();
    test_chat_template_multiple_specials();
    test_chat_template_only_specials();
    test_chat_template_large_text_roundtrip();
    test_with_specials_custom();
    test_custom_tokenizer();
    test_large_text_roundtrip();
    test_tensor_to_text_empty_tensor();
    test_text_to_tensor_null_empty();

    printf("  ALL PASS\n");
    dnn_ctx_destroy(&ctx);

    return 0;
}
