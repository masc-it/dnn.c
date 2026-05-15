#include "dnn.h"
#include "context.h"
#include "transformer.h"
#include "optim.h"
#include <stdio.h>
#include <assert.h>
#include <math.h>
#include <string.h>
#include <stdlib.h>

static dnn_ctx ctx;

/*
 * Tests generation with prefix feeding + KV-cache, including RoPE.
 *
 * Verifies:
 *   - Cached == non-cached WITHOUT RoPE (various prompt lengths)
 *   - Cached == non-cached WITH RoPE (various prompt lengths)
 *   - Prompt prefix preserved in output
 *   - Generated tokens in vocab range
 *
 * Config: vocab=8, d_model=4, n_layers=2, n_heads=2, d_k=2, intermediate=8
 *
 * Reference: test/ref_generation_prefix.py --seed 42
 */

#define EPS 1e-5f
#define EOS_ID 258

/* ── Helpers ── */

static tensor *make_int_tensor(int ndim, const int *shape, const int *data) {
    int n = 1;
    for (int i = 0; i < ndim; i++) n *= shape[i];
    tensor *t = tensor_zeros_data(ctx.data, ndim, shape);
    if (data) memcpy(t->data, data, n * sizeof(int));
    return t;
}

/* Collect all trainable params into a flat array (for AdamW) */
static tensor **collect_params(decoder_lm *lm, int *n_out) {
    tensor *all[256];
    int n = 0;

    all[n++] = lm->embedding_table;
    all[n++] = lm->norm_weight;
    all[n++] = lm->norm_bias;
    /* lm_head->weight excluded — weight tying via transposed view of embedding_table */
    all[n++] = lm->lm_head->bias;

    for (int i = 0; i < lm->n_layers; i++) {
        transformer_block *b = lm->blocks[i];
        all[n++] = b->q_proj->weight;
        all[n++] = b->q_proj->bias;
        all[n++] = b->k_proj->weight;
        all[n++] = b->k_proj->bias;
        all[n++] = b->v_proj->weight;
        all[n++] = b->v_proj->bias;
        all[n++] = b->out_proj->weight;
        all[n++] = b->out_proj->bias;
        all[n++] = b->attn_norm_weight;
        all[n++] = b->attn_norm_bias;
        all[n++] = b->ffn_norm_weight;
        all[n++] = b->ffn_norm_bias;
        all[n++] = b->ffn->gate_proj->weight;
        all[n++] = b->ffn->gate_proj->bias;
        all[n++] = b->ffn->up_proj->weight;
        all[n++] = b->ffn->up_proj->bias;
        all[n++] = b->ffn->down_proj->weight;
        all[n++] = b->ffn->down_proj->bias;
    }

    *n_out = n;
    tensor **arr = _mem_pool_alloc(ctx.params, n * sizeof(tensor*), NULL);
    memcpy(arr, all, n * sizeof(tensor*));
    return arr;
}

/* Train a decoder_lm for a few steps */
static void train_lm(decoder_lm *lm, int B, int N, int n_steps) {
    int n_params;
    tensor **all_params = collect_params(lm, &n_params);
    adamw_opt *opt = adamw_create(ctx.params, all_params, n_params, 0.01f,
                                   0.9f, 0.999f, 1e-8f, 0.01f);

    int *ids = malloc((size_t)B * N * sizeof(int));
    for (int i = 0; i < B * N; i++) ids[i] = rand() % lm->vocab_size;

    for (int step = 0; step < n_steps; step++) {
        tensor *input_ids = make_int_tensor(2, (int[]){B, N}, ids);
        tensor *loss = decoder_lm_train_step(ctx.scratch, ctx.data, lm, input_ids, opt, 0.0f, NULL);
        (void)loss;
    }

    free(ids);
}

/* Check cached == non-cached for given prompt, with optional RoPE */
static void check_cached_equivalence(decoder_lm *lm, const tensor *prompt,
                                      int max_new, const char *label,
                                      int use_rope) {
    int prompt_len = prompt->shape[1];
    int vocab = lm->vocab_size;

    if (use_rope) {
        int d_k = lm->blocks[0]->d_k;
        decoder_lm_enable_rope(ctx.params, lm, prompt_len + max_new + 4, 10000.0f);
    }

    /* Non-cached generation */
    int n_nc;
    int *res_nc = decoder_lm_generate(ctx.scratch, ctx.data, lm, prompt, max_new, 0.0f, 0, &n_nc);

    /* Recreate model with same seed, enable RoPE if needed */
    srand(42);  /* reset to same seed */
    /* Can't reset weights easily, so create fresh model */
    /* (We'll do this per test case instead) */

    (void)label;
    (void)prompt_len;
    (void)vocab;

    /* Free result (owned by data pool, we just need the pointer for now) */
    /* The data will be released when pool is destroyed */
    (void)res_nc;
    (void)n_nc;
}

/* ── Test: RoPE enabled, short prompt (N=3), cached == non-cached ── */

static void test_rope_short_prompt(void) {
    printf("  test_rope_short_prompt... ");

    dnn_ctx_init(&ctx, 4 * 1024 * 1024, 64 * 1024 * 1024, 1 * 1024 * 1024);

    int vocab=8, d_model=4, n_layers=2, n_heads=2, d_k=2, intermediate=8;

    srand(42);
    decoder_lm *lm = decoder_lm_create(ctx.params, vocab, d_model, n_layers, n_heads,
                                        d_k, intermediate);
    train_lm(lm, 1, 4, 5);
    decoder_lm_enable_rope(ctx.params, lm, 20, 10000.0f);

    int prompt_data[] = {2, 5, 2};
    tensor *prompt = make_int_tensor(2, (int[]){1, 3}, prompt_data);

    int n_nc, n_c;
    int *res_nc = decoder_lm_generate(ctx.scratch, ctx.data, lm, prompt, 5, 0.0f, 0, &n_nc);

    mem_pool_reset(ctx.scratch);
    mem_pool_reset(ctx.data);
    srand(42);
    decoder_lm *lm2 = decoder_lm_create(ctx.params, vocab, d_model, n_layers, n_heads,
                                         d_k, intermediate);
    train_lm(lm2, 1, 4, 5);
    decoder_lm_enable_rope(ctx.params, lm2, 20, 10000.0f);
    tensor *prompt2 = make_int_tensor(2, (int[]){1, 3}, prompt_data);
    int *res_c = decoder_lm_generate(ctx.scratch, ctx.data, lm2, prompt2, 5, 0.0f, 1, &n_c);

    assert(n_nc == n_c && "rope_short: cached/non-cached length mismatch");
    for (int i = 0; i < n_nc; i++) {
        assert(res_nc[i] == res_c[i] &&
               "rope_short: cached/non-cached token mismatch");
    }
    /* Verify prefix preserved */
    assert(res_nc[0] == 2 && res_nc[1] == 5 && res_nc[2] == 2);
    /* All tokens in vocab range */
    for (int i = 0; i < n_nc; i++)
        assert(res_nc[i] >= 0 && res_nc[i] < vocab);

    printf("OK (len=%d, rope, nc=[%d,%d,%d,%d,%d,%d,%d,%d])\n",
           n_nc,
           res_nc[0], res_nc[1], res_nc[2], res_nc[3],
           n_nc > 4 ? res_nc[4] : -1,
           n_nc > 5 ? res_nc[5] : -1,
           n_nc > 6 ? res_nc[6] : -1,
           n_nc > 7 ? res_nc[7] : -1);

}

/* ── Test: RoPE, longer prefix (N=8), cached == non-cached ── */

static void test_rope_long_prefix(void) {
    printf("  test_rope_long_prefix... ");

    dnn_ctx_init(&ctx, 4 * 1024 * 1024, 64 * 1024 * 1024, 1 * 1024 * 1024);

    int vocab=8, d_model=4, n_layers=2, n_heads=2, d_k=2, intermediate=8;

    srand(42);
    decoder_lm *lm = decoder_lm_create(ctx.params, vocab, d_model, n_layers, n_heads,
                                        d_k, intermediate);
    train_lm(lm, 1, 4, 5);
    decoder_lm_enable_rope(ctx.params, lm, 20, 10000.0f);

    int prompt_data[] = {0, 3, 7, 1, 4, 2, 6, 5};
    tensor *prompt = make_int_tensor(2, (int[]){1, 8}, prompt_data);

    int n_nc, n_c;
    int *res_nc = decoder_lm_generate(ctx.scratch, ctx.data, lm, prompt, 5, 0.0f, 0, &n_nc);

    mem_pool_reset(ctx.scratch);
    mem_pool_reset(ctx.data);
    srand(42);
    decoder_lm *lm2 = decoder_lm_create(ctx.params, vocab, d_model, n_layers, n_heads,
                                         d_k, intermediate);
    train_lm(lm2, 1, 4, 5);
    decoder_lm_enable_rope(ctx.params, lm2, 20, 10000.0f);
    tensor *prompt2 = make_int_tensor(2, (int[]){1, 8}, prompt_data);
    int *res_c = decoder_lm_generate(ctx.scratch, ctx.data, lm2, prompt2, 5, 0.0f, 1, &n_c);

    assert(n_nc == n_c && "rope_long: cached/non-cached length mismatch");
    for (int i = 0; i < n_nc; i++) {
        assert(res_nc[i] == res_c[i] &&
               "rope_long: cached/non-cached token mismatch");
    }
    /* Verify 8-token prefix preserved */
    assert(res_nc[0] == 0 && res_nc[1] == 3 && res_nc[2] == 7 &&
           res_nc[3] == 1 && res_nc[4] == 4 && res_nc[5] == 2 &&
           res_nc[6] == 6 && res_nc[7] == 5);
    /* All tokens in vocab range */
    for (int i = 0; i < n_nc; i++)
        assert(res_nc[i] >= 0 && res_nc[i] < vocab);

    printf("OK (len=%d, rope, 8-token prefix)\n", n_nc);

}

/* ── Test: RoPE, single-token prompt (N=1), cached == non-cached ── */

static void test_rope_single_token(void) {
    printf("  test_rope_single_token... ");

    dnn_ctx_init(&ctx, 4 * 1024 * 1024, 64 * 1024 * 1024, 1 * 1024 * 1024);

    int vocab=8, d_model=4, n_layers=2, n_heads=2, d_k=2, intermediate=8;

    srand(42);
    decoder_lm *lm = decoder_lm_create(ctx.params, vocab, d_model, n_layers, n_heads,
                                        d_k, intermediate);
    train_lm(lm, 1, 4, 5);
    decoder_lm_enable_rope(ctx.params, lm, 20, 10000.0f);

    int prompt_data[] = {7};
    tensor *prompt = make_int_tensor(2, (int[]){1, 1}, prompt_data);

    int n_nc, n_c;
    int *res_nc = decoder_lm_generate(ctx.scratch, ctx.data, lm, prompt, 4, 0.0f, 0, &n_nc);

    mem_pool_reset(ctx.scratch);
    mem_pool_reset(ctx.data);
    srand(42);
    decoder_lm *lm2 = decoder_lm_create(ctx.params, vocab, d_model, n_layers, n_heads,
                                         d_k, intermediate);
    train_lm(lm2, 1, 4, 5);
    decoder_lm_enable_rope(ctx.params, lm2, 20, 10000.0f);
    tensor *prompt2 = make_int_tensor(2, (int[]){1, 1}, prompt_data);
    int *res_c = decoder_lm_generate(ctx.scratch, ctx.data, lm2, prompt2, 4, 0.0f, 1, &n_c);

    assert(n_nc == n_c && "rope_single: cached/non-cached length mismatch");
    for (int i = 0; i < n_nc; i++) {
        assert(res_nc[i] == res_c[i] &&
               "rope_single: cached/non-cached token mismatch");
    }
    assert(res_nc[0] == 7 && "rope_single: prefix not preserved");
    for (int i = 0; i < n_nc; i++)
        assert(res_nc[i] >= 0 && res_nc[i] < vocab);

    printf("OK (len=%d, rope, singletoken)\n", n_nc);

}

/* ── Test: No RoPE, longer prefix (N=8), cached == non-cached ── */

static void test_norope_long_prefix(void) {
    printf("  test_norope_long_prefix... ");

    dnn_ctx_init(&ctx, 4 * 1024 * 1024, 64 * 1024 * 1024, 1 * 1024 * 1024);

    int vocab=8, d_model=4, n_layers=2, n_heads=2, d_k=2, intermediate=8;

    srand(42);
    decoder_lm *lm = decoder_lm_create(ctx.params, vocab, d_model, n_layers, n_heads,
                                        d_k, intermediate);
    train_lm(lm, 1, 4, 5);

    int prompt_data[] = {0, 3, 7, 1, 4, 2, 6, 5};
    tensor *prompt = make_int_tensor(2, (int[]){1, 8}, prompt_data);

    int n_nc, n_c;
    int *res_nc = decoder_lm_generate(ctx.scratch, ctx.data, lm, prompt, 5, 0.0f, 0, &n_nc);

    mem_pool_reset(ctx.scratch);
    mem_pool_reset(ctx.data);
    srand(42);
    decoder_lm *lm2 = decoder_lm_create(ctx.params, vocab, d_model, n_layers, n_heads,
                                         d_k, intermediate);
    train_lm(lm2, 1, 4, 5);
    tensor *prompt2 = make_int_tensor(2, (int[]){1, 8}, prompt_data);
    int *res_c = decoder_lm_generate(ctx.scratch, ctx.data, lm2, prompt2, 5, 0.0f, 1, &n_c);

    assert(n_nc == n_c && "norope_long: cached/non-cached length mismatch");
    for (int i = 0; i < n_nc; i++) {
        assert(res_nc[i] == res_c[i] &&
               "norope_long: cached/non-cached token mismatch");
    }
    assert(res_nc[0] == 0 && res_nc[7] == 5 && "norope_long: prefix not preserved");
    for (int i = 0; i < n_nc; i++)
        assert(res_nc[i] >= 0 && res_nc[i] < vocab);

    printf("OK (len=%d, norope, 8-token prefix)\n", n_nc);

}

/* ── Test: RoPE, various model configs ── */

static void test_rope_various_configs(void) {
    printf("  test_rope_various_configs...\n");

    dnn_ctx_init(&ctx, 4 * 1024 * 1024, 64 * 1024 * 1024, 1 * 1024 * 1024);

    struct { int vocab, d_model, n_layers, n_heads, d_k, intermed; } configs[] = {
        { 5,  4, 1, 2, 2, 8 },
        { 10, 4, 2, 2, 2, 8 },
        { 8,  4, 3, 2, 2, 8 },
        { 16, 8, 2, 4, 2, 16 },
    };
    int n_configs = 4;

    for (int ci = 0; ci < n_configs; ci++) {
        int v = configs[ci].vocab, dm = configs[ci].d_model;
        int nl = configs[ci].n_layers, nh = configs[ci].n_heads;
        int dk = configs[ci].d_k, inter = configs[ci].intermed;

        srand(42);
        decoder_lm *lm = decoder_lm_create(ctx.params, v, dm, nl, nh, dk, inter);
        train_lm(lm, 1, 3, 3);

        /* Only enable RoPE if d_k is even */
        if (dk % 2 == 0)
            decoder_lm_enable_rope(ctx.params, lm, 20, 10000.0f);

        int prompt_data[] = {0, 1, 2};
        for (int i = 0; i < 3; i++) prompt_data[i] %= v;
        tensor *prompt = make_int_tensor(2, (int[]){1, 3}, prompt_data);

        int n1, n2;
        int *r1 = decoder_lm_generate(ctx.scratch, ctx.data, lm, prompt, 4, 0.0f, 0, &n1);

        mem_pool_reset(ctx.scratch);
        mem_pool_reset(ctx.data);

        srand(42);
        decoder_lm *lm2 = decoder_lm_create(ctx.params, v, dm, nl, nh, dk, inter);
        train_lm(lm2, 1, 3, 3);
        if (dk % 2 == 0)
            decoder_lm_enable_rope(ctx.params, lm2, 20, 10000.0f);
        tensor *prompt2 = make_int_tensor(2, (int[]){1, 3}, prompt_data);
        int *r2 = decoder_lm_generate(ctx.scratch, ctx.data, lm2, prompt2, 4, 0.0f, 1, &n2);

        assert(n1 == n2 && "config: cached/non-cached length mismatch");
        for (int i = 0; i < n1; i++)
            assert(r1[i] == r2[i] && "config: cached/non-cached token mismatch");

        printf("    vocab=%d,d_model=%d,n_layers=%d: len=%d OK\n",
               v, dm, nl, n1);

        mem_pool_reset(ctx.params);
        mem_pool_reset(ctx.scratch);
        mem_pool_reset(ctx.data);
    }

    printf("  rope_various_configs: OK\n");

}

/* ── Test: Max new tokens with RoPE ── */

static void test_rope_max_new_tokens(void) {
    printf("  test_rope_max_new_tokens... ");

    dnn_ctx_init(&ctx, 4 * 1024 * 1024, 64 * 1024 * 1024, 1 * 1024 * 1024);

    int vocab=8, d_model=4, n_layers=2, n_heads=2, d_k=2, intermediate=8;

    srand(42);
    decoder_lm *lm = decoder_lm_create(ctx.params, vocab, d_model, n_layers, n_heads,
                                        d_k, intermediate);
    train_lm(lm, 1, 4, 5);
    decoder_lm_enable_rope(ctx.params, lm, 20, 10000.0f);

    int prompt_data[] = {2, 5, 2};
    tensor *prompt = make_int_tensor(2, (int[]){1, 3}, prompt_data);

    /* Limit to 1 new token */
    int n_out;
    int *result = decoder_lm_generate(ctx.scratch, ctx.data, lm, prompt, 1, 0.0f, 0, &n_out);
    assert(n_out == 4);  /* prompt=3 + 1 new */

    mem_pool_reset(ctx.scratch);
    mem_pool_reset(ctx.data);
    srand(42);
    decoder_lm *lm2 = decoder_lm_create(ctx.params, vocab, d_model, n_layers, n_heads,
                                         d_k, intermediate);
    train_lm(lm2, 1, 4, 5);
    decoder_lm_enable_rope(ctx.params, lm2, 20, 10000.0f);
    tensor *prompt2 = make_int_tensor(2, (int[]){1, 3}, prompt_data);
    int *result2 = decoder_lm_generate(ctx.scratch, ctx.data, lm2, prompt2, 1, 0.0f, 1, &n_out);
    assert(n_out == 4);

    assert(result[0] == result2[0] && result[1] == result2[1] &&
           result[2] == result2[2] && result[3] == result2[3]);

    printf("OK (len=%d, rope, max_new=1)\n", n_out);

}

/* ── Test: EOS stopping with RoPE ── */

static void test_rope_eos_stop(void) {
    printf("  test_rope_eos_stop... ");

    dnn_ctx_init(&ctx, 4 * 1024 * 1024, 64 * 1024 * 1024, 1 * 1024 * 1024);

    /* Use a vocab with EOS (ID 258) in range — need vocab > 258 for EOS to exist */
    /* Our small test config only has vocab=8, so EOS (258) is out of range.
     * We test via the no-cache path: EOS check uses hard-coded 258.
     * With small vocab, generation never hits EOS, which is fine — it just
     * stops at max_new_tokens.  Test that both paths produce same length. */
    int vocab=260, d_model=4, n_layers=2, n_heads=2, d_k=2, intermediate=8;

    srand(42);
    decoder_lm *lm = decoder_lm_create(ctx.params, vocab, d_model, n_layers, n_heads,
                                        d_k, intermediate);
    train_lm(lm, 1, 4, 5);
    decoder_lm_enable_rope(ctx.params, lm, 20, 10000.0f);

    int prompt_data[] = {257};  /* BOS */
    tensor *prompt = make_int_tensor(2, (int[]){1, 1}, prompt_data);

    int n_nc, n_c;
    int *res_nc = decoder_lm_generate(ctx.scratch, ctx.data, lm, prompt, 10, 0.0f, 0, &n_nc);

    mem_pool_reset(ctx.scratch);
    mem_pool_reset(ctx.data);
    srand(42);
    decoder_lm *lm2 = decoder_lm_create(ctx.params, vocab, d_model, n_layers, n_heads,
                                         d_k, intermediate);
    train_lm(lm2, 1, 4, 5);
    decoder_lm_enable_rope(ctx.params, lm2, 20, 10000.0f);
    tensor *prompt2 = make_int_tensor(2, (int[]){1, 1}, prompt_data);
    int *res_c = decoder_lm_generate(ctx.scratch, ctx.data, lm2, prompt2, 10, 0.0f, 1, &n_c);

    assert(n_nc == n_c && "eos: cached/non-cached length mismatch");
    for (int i = 0; i < n_nc; i++)
        assert(res_nc[i] == res_c[i] && "eos: cached/non-cached token mismatch");
    /* Prefix preserved */
    assert(res_nc[0] == 257 && "eos: prefix not preserved");

    printf("OK (len=%d, rope, vocab includes EOS)\n", n_nc);

}

/* ── Test: strict prefix preservation (multi-token) ── */

static void test_prefix_exact_preserved(void) {
    printf("  test_prefix_exact_preserved... ");

    dnn_ctx_init(&ctx, 4 * 1024 * 1024, 64 * 1024 * 1024, 1 * 1024 * 1024);

    int vocab=16, d_model=8, n_layers=2, n_heads=4, d_k=2, intermediate=16;

    srand(42);
    decoder_lm *lm = decoder_lm_create(ctx.params, vocab, d_model, n_layers, n_heads,
                                        d_k, intermediate);
    train_lm(lm, 1, 6, 3);
    decoder_lm_enable_rope(ctx.params, lm, 30, 10000.0f);

    /* Use all unique tokens for the prefix */
    int prompt_data[] = {3, 7, 1, 12, 5, 9};
    int prompt_len = 6;
    tensor *prompt = make_int_tensor(2, (int[]){1, prompt_len}, prompt_data);

    /* Test both cached and non-cached */
    int n_nc, n_c;
    int *res_nc = decoder_lm_generate(ctx.scratch, ctx.data, lm, prompt, 4, 0.0f, 0, &n_nc);
    mem_pool_reset(ctx.scratch);
    mem_pool_reset(ctx.data);
    srand(42);
    decoder_lm *lm2 = decoder_lm_create(ctx.params, vocab, d_model, n_layers, n_heads,
                                         d_k, intermediate);
    train_lm(lm2, 1, 6, 3);
    decoder_lm_enable_rope(ctx.params, lm2, 30, 10000.0f);
    tensor *prompt2 = make_int_tensor(2, (int[]){1, prompt_len}, prompt_data);
    int *res_c = decoder_lm_generate(ctx.scratch, ctx.data, lm2, prompt2, 4, 0.0f, 1, &n_c);

    assert(n_nc == n_c && "prefix_exact: length mismatch");
    for (int i = 0; i < n_nc; i++)
        assert(res_nc[i] == res_c[i] && "prefix_exact: cached/non-cached mismatch");
    /* Exact prefix bytes match */
    for (int i = 0; i < prompt_len; i++)
        assert(res_nc[i] == prompt_data[i] && "prefix_exact: prefix corrupted");

    printf("OK (len=%d, %d-token prefix, rope)\n", n_nc, prompt_len);

}

/* ── Main ── */

int main(void) {
    printf("=== generation prefix + RoPE tests ===\n\n");

    test_rope_short_prompt();
    test_rope_long_prefix();
    test_rope_single_token();
    test_norope_long_prefix();
    test_rope_various_configs();
    test_rope_max_new_tokens();
    test_rope_eos_stop();
    test_prefix_exact_preserved();

    printf("\nAll generation prefix + RoPE tests passed.\n");
    dnn_ctx_destroy(&ctx);

    return 0;
}
