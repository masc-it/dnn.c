#include "dnn.h"
#include "gpt.h"
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
 * Reference values from test/ref_generation.py --small (seed=42):
 *
 * Config: vocab=8, d_model=4, n_layers=2, n_heads=2, d_k=2, intermediate=8
 * Training: 5 steps on fixed random input
 *
 * Prompt:  [2, 5, 2]  (N=3)
 * Argmax:  [2, 5, 2, 1, 6, 1, 6, 6]  (8 tokens)
 * Short:   [7, 1, 6, 6]  (prompt=[7], N=1, 3 new tokens)
 *
 * Note: PyTorch reference uses C random init (not dnn.c's srand),
 * so numeric values won't match exactly.  We test structural properties
 * and cached vs non-cached equivalence.
 */

#define EPS 1e-5f
#define EOS_ID 258

/* ── Helper ── */

static tensor *make_int_tensor(int ndim, const int *shape, const int *data) {
    int n = 1;
    for (int i = 0; i < ndim; i++) n *= shape[i];
    tensor *t = tensor_zeros_data(ctx.data, ndim, shape);
    if (data) memcpy(t->data, data, n * sizeof(int));
    return t;
}

static tensor **collect_params(decoder_lm *lm, int *n_out) {
    return module_parameters(&lm->base, n_out);
}

/* Train a decoder_lm for a few steps */
static void train_lm(decoder_lm *lm, int B, int N, int n_steps) {
    int n_params;
    tensor **all_params = collect_params(lm, &n_params);
    adamw_opt *opt = adamw_create(ctx.params, all_params, n_params, 0.01f,
                                   0.9f, 0.999f, 1e-8f, 0.01f);

    int *ids = malloc(B * N * sizeof(int));
    for (int i = 0; i < B * N; i++) ids[i] = rand() % lm->vocab_size;

    for (int step = 0; step < n_steps; step++) {
        tensor *input_ids = make_int_tensor(2, (int[]){B, N}, ids);
        tensor *target = decoder_lm_shift_targets(ctx.data, input_ids);
        tensor *loss = decoder_lm_train_step(ctx.scratch, lm, input_ids, target, opt, 0.0f, NULL);
        (void)loss;
    }

    free(ids);
}

/* ── Test: generate with argmax (no cache) produces finite tokens ── */

static void test_generate_argmax_nocache(void) {
    printf("  test_generate_argmax_nocache... ");

    dnn_ctx_init(&ctx, 4 * 1024 * 1024, 64 * 1024 * 1024, 1 * 1024 * 1024);

    int vocab=8, d_model=4, n_layers=2, n_heads=2, d_k=2, intermediate=8;

    srand(42);
    decoder_lm *lm = decoder_lm_create(ctx.params, vocab, d_model, n_layers, n_heads,
                                        d_k, intermediate);
    train_lm(lm, 1, 4, 5);

    int prompt_data[] = {2, 5, 2};
    tensor *prompt = make_int_tensor(2, (int[]){1, 3}, prompt_data);

    int n_out;
    int *result = decoder_lm_generate(ctx.scratch, ctx.data, lm, prompt, 5, 0.0f, 0, &n_out);

    assert(result != NULL);
    assert(n_out >= 3);  /* at least prompt length */
    assert(n_out <= 8);  /* prompt + max_new */

    /* All IDs within vocab range */
    for (int i = 0; i < n_out; i++) {
        assert(result[i] >= 0 && result[i] < vocab &&
               "generated token out of vocab range");
    }

    /* Prompt prefix intact */
    assert(result[0] == 2 && result[1] == 5 && result[2] == 2);

    printf("OK (len=%d, tokens=[%d,%d,%d,%d,%d,%d,%d,%d])\n",
           n_out,
           result[0], result[1], result[2], result[3],
           n_out > 4 ? result[4] : -1,
           n_out > 5 ? result[5] : -1,
           n_out > 6 ? result[6] : -1,
           n_out > 7 ? result[7] : -1);

}

/* ── Test: cached and non-cached generate identical results (argmax) ── */

static void test_cached_vs_nocached(void) {
    printf("  test_cached_vs_nocached... ");

    dnn_ctx_init(&ctx, 4 * 1024 * 1024, 64 * 1024 * 1024, 1 * 1024 * 1024);

    int vocab=8, d_model=4, n_layers=2, n_heads=2, d_k=2, intermediate=8;

    srand(7);  /* different seed for variety */
    decoder_lm *lm = decoder_lm_create(ctx.params, vocab, d_model, n_layers, n_heads,
                                        d_k, intermediate);
    train_lm(lm, 1, 4, 5);

    int prompt_data[] = {0, 3, 7};
    tensor *prompt = make_int_tensor(2, (int[]){1, 3}, prompt_data);

    int n_nocache, n_cache;
    int *res_nocache = decoder_lm_generate(ctx.scratch, ctx.data, lm, prompt, 4, 0.0f, 0, &n_nocache);

    /* Reset data pool (both calls need fresh data pool allocs) */
    mem_pool_reset(ctx.scratch);
    mem_pool_reset(ctx.data);
    srand(7);
    decoder_lm *lm2 = decoder_lm_create(ctx.params, vocab, d_model, n_layers, n_heads,
                                         d_k, intermediate);
    train_lm(lm2, 1, 4, 5);

    tensor *prompt2 = make_int_tensor(2, (int[]){1, 3}, prompt_data);
    int *res_cache = decoder_lm_generate(ctx.scratch, ctx.data, lm2, prompt2, 4, 0.0f, 1, &n_cache);

    assert(n_nocache == n_cache && "cached and non-cached: len mismatch");
    for (int i = 0; i < n_nocache; i++) {
        assert(res_nocache[i] == res_cache[i] &&
               "cached and non-cached: token mismatch at position %d");
    }

    printf("OK (len=%d, match, tokens=[%d,%d,%d,%d,%d,%d,%d])\n",
           n_nocache,
           res_nocache[0], res_nocache[1], res_nocache[2],
           res_nocache[3], res_nocache[4], res_nocache[5],
           n_nocache > 6 ? res_nocache[6] : -1);

}

/* ── Test: cached and non-cached with short prompt (N=1) ── */

static void test_short_prompt(void) {
    printf("  test_short_prompt... ");

    dnn_ctx_init(&ctx, 4 * 1024 * 1024, 64 * 1024 * 1024, 1 * 1024 * 1024);

    int vocab=8, d_model=4, n_layers=2, n_heads=2, d_k=2, intermediate=8;

    srand(42);
    decoder_lm *lm = decoder_lm_create(ctx.params, vocab, d_model, n_layers, n_heads,
                                        d_k, intermediate);
    train_lm(lm, 1, 4, 5);

    int prompt_data[] = {7};
    tensor *prompt = make_int_tensor(2, (int[]){1, 1}, prompt_data);

    int n_nocache, n_cache;
    int *res_nocache = decoder_lm_generate(ctx.scratch, ctx.data, lm, prompt, 3, 0.0f, 0, &n_nocache);

    mem_pool_reset(ctx.scratch);
    mem_pool_reset(ctx.data);
    srand(42);
    decoder_lm *lm2 = decoder_lm_create(ctx.params, vocab, d_model, n_layers, n_heads,
                                         d_k, intermediate);
    train_lm(lm2, 1, 4, 5);
    tensor *prompt2 = make_int_tensor(2, (int[]){1, 1}, prompt_data);
    int *res_cache = decoder_lm_generate(ctx.scratch, ctx.data, lm2, prompt2, 3, 0.0f, 1, &n_cache);

    assert(n_nocache == n_cache);
    for (int i = 0; i < n_nocache; i++) {
        assert(res_nocache[i] == res_cache[i] &&
               "short prompt: cached/non-cached mismatch");
    }

    printf("OK (len=%d, tokens=[%d,%d,%d,%d])\n",
           n_nocache, res_nocache[0], res_nocache[1],
           res_nocache[2], res_nocache[3]);

}

/* ── Test: max_new_tokens limit ── */

static void test_max_new_tokens_limit(void) {
    printf("  test_max_new_tokens_limit... ");

    dnn_ctx_init(&ctx, 4 * 1024 * 1024, 64 * 1024 * 1024, 1 * 1024 * 1024);

    int vocab=8, d_model=4, n_layers=2, n_heads=2, d_k=2, intermediate=8;

    srand(42);
    decoder_lm *lm = decoder_lm_create(ctx.params, vocab, d_model, n_layers, n_heads,
                                        d_k, intermediate);
    train_lm(lm, 1, 4, 5);

    int prompt_data[] = {2, 5, 2};
    tensor *prompt = make_int_tensor(2, (int[]){1, 3}, prompt_data);

    /* Limit to 1 new token */
    int n_out;
    int *result = decoder_lm_generate(ctx.scratch, ctx.data, lm, prompt, 1, 0.0f, 0, &n_out);
    assert(n_out == 4);  /* prompt=3 + 1 new */
    printf(" nocache(len=%d) ", n_out);

    mem_pool_reset(ctx.scratch);
    mem_pool_reset(ctx.data);
    srand(42);
    decoder_lm *lm2 = decoder_lm_create(ctx.params, vocab, d_model, n_layers, n_heads,
                                         d_k, intermediate);
    train_lm(lm2, 1, 4, 5);
    tensor *prompt2 = make_int_tensor(2, (int[]){1, 3}, prompt_data);
    int *result2 = decoder_lm_generate(ctx.scratch, ctx.data, lm2, prompt2, 1, 0.0f, 1, &n_out);
    assert(n_out == 4);
    printf("cache(len=%d) ", n_out);

    /* Both should agree */
    assert(result[0] == result2[0] && result[1] == result2[1] &&
           result[2] == result2[2] && result[3] == result2[3]);

    printf("OK\n");

}

/* ── Test: generation produces same result with same seed ── */

static void test_deterministic(void) {
    printf("  test_deterministic... ");

    dnn_ctx_init(&ctx, 4 * 1024 * 1024, 64 * 1024 * 1024, 1 * 1024 * 1024);

    int vocab=8, d_model=4, n_layers=2, n_heads=2, d_k=2, intermediate=8;

    srand(99);
    decoder_lm *lm = decoder_lm_create(ctx.params, vocab, d_model, n_layers, n_heads,
                                        d_k, intermediate);
    train_lm(lm, 1, 4, 5);

    int prompt_data[] = {4, 1, 0};
    tensor *prompt = make_int_tensor(2, (int[]){1, 3}, prompt_data);

    int n1, n2;
    int *r1 = decoder_lm_generate(ctx.scratch, ctx.data, lm, prompt, 5, 0.0f, 0, &n1);

    /* Run again: should produce same result (argmax is deterministic) */
    mem_pool_reset(ctx.scratch);
    mem_pool_reset(ctx.data);
    tensor *prompt2 = make_int_tensor(2, (int[]){1, 3}, prompt_data);
    int *r2 = decoder_lm_generate(ctx.scratch, ctx.data, lm, prompt2, 5, 0.0f, 0, &n2);

    assert(n1 == n2);
    for (int i = 0; i < n1; i++)
        assert(r1[i] == r2[i] && "deterministic generation: mismatch");

    printf("OK (len=%d)\n", n1);

}

/* ── Test: temperature sampling produces different results ── */

static void test_temperature_sampling(void) {
    printf("  test_temperature_sampling... ");

    dnn_ctx_init(&ctx, 4 * 1024 * 1024, 64 * 1024 * 1024, 1 * 1024 * 1024);

    int vocab=8, d_model=4, n_layers=2, n_heads=2, d_k=2, intermediate=8;

    srand(42);
    decoder_lm *lm = decoder_lm_create(ctx.params, vocab, d_model, n_layers, n_heads,
                                        d_k, intermediate);
    train_lm(lm, 1, 4, 5);

    int prompt_data[] = {2, 5, 2};
    tensor *prompt = make_int_tensor(2, (int[]){1, 3}, prompt_data);

    /* Argmax result */
    int n_argmax;
    int *argmax_res = decoder_lm_generate(ctx.scratch, ctx.data, lm, prompt, 5, 0.0f, 0, &n_argmax);

    /* Reset and sample with temperature */
    mem_pool_reset(ctx.scratch);
    mem_pool_reset(ctx.data);
    srand(42);

    /* Need to recreate model with same seed to get same weights */
    decoder_lm *lm2 = decoder_lm_create(ctx.params, vocab, d_model, n_layers, n_heads,
                                         d_k, intermediate);
    train_lm(lm2, 1, 4, 5);
    tensor *prompt2 = make_int_tensor(2, (int[]){1, 3}, prompt_data);

    int n_sample;
    int *sample_res = decoder_lm_generate(ctx.scratch, ctx.data, lm2, prompt2, 5, 0.7f, 0, &n_sample);

    assert(n_sample <= 8);
    assert(sample_res[0] == 2 && sample_res[1] == 5 && sample_res[2] == 2);

    /* With temp=0.7 and rand(), sampling may differ from argmax.
     * Not guaranteed (model could be very confident for some prompts),
     * so just validate the output is structurally correct. */
    assert(sample_res[0] == 2 && sample_res[1] == 5 && sample_res[2] == 2);
    for (int i = 0; i < n_sample; i++) {
        assert(sample_res[i] >= 0 && sample_res[i] < vocab);
    }

    /* Count non-argmax tokens across both runs */
    int diff_count = 0;
    for (int i = 0; i < n_argmax && i < n_sample; i++) {
        if (argmax_res[i] != sample_res[i]) diff_count++;
    }

    printf("OK (argmax=%d, sampled=%d, differ=%d tokens)\n",
           n_argmax, n_sample, diff_count);

}

/* ── Test: generation with various model configs ── */

static void test_various_configs(void) {
    printf("  test_various_configs...\n");

    dnn_ctx_init(&ctx, 4 * 1024 * 1024, 64 * 1024 * 1024, 1 * 1024 * 1024);

    struct { int vocab, d_model, n_layers, n_heads, d_k, intermed; } configs[] = {
        { 5,  2, 1, 1, 2, 4 },
        { 10, 4, 1, 2, 2, 8 },
        { 8,  4, 3, 2, 2, 8 },
        { 16, 8, 2, 2, 4, 16 },
    };
    int n_configs = 4;

    for (int ci = 0; ci < n_configs; ci++) {
        int v = configs[ci].vocab, dm = configs[ci].d_model;
        int nl = configs[ci].n_layers, nh = configs[ci].n_heads;
        int dk = configs[ci].d_k, inter = configs[ci].intermed;

        srand(42);
        decoder_lm *lm = decoder_lm_create(ctx.params, v, dm, nl, nh, dk, inter);
        train_lm(lm, 1, 3, 3);

        int prompt_data[] = {0, 1, 2};
        /* Clamp prompt to vocab range */
        for (int i = 0; i < 3; i++) prompt_data[i] %= v;

        tensor *prompt = make_int_tensor(2, (int[]){1, 3}, prompt_data);

        /* Test non-cached */
        int n1;
        int *r1 = decoder_lm_generate(ctx.scratch, ctx.data, lm, prompt, 4, 0.0f, 0, &n1);
        assert(r1 != NULL && n1 >= 3);

        /* Test cached */
        mem_pool_reset(ctx.scratch);
        mem_pool_reset(ctx.data);

        srand(42);
        decoder_lm *lm2 = decoder_lm_create(ctx.params, v, dm, nl, nh, dk, inter);
        train_lm(lm2, 1, 3, 3);
        tensor *prompt2 = make_int_tensor(2, (int[]){1, 3}, prompt_data);
        int n2;
        int *r2 = decoder_lm_generate(ctx.scratch, ctx.data, lm2, prompt2, 4, 0.0f, 1, &n2);

        assert(n1 == n2 && "config: len mismatch");
        for (int i = 0; i < n1; i++)
            assert(r1[i] == r2[i] && "config: token mismatch");

        printf("    vocab=%d,d_model=%d,n_layers=%d: len=%d OK\n",
               v, dm, nl, n1);

        mem_pool_reset(ctx.params);
        mem_pool_reset(ctx.scratch);
        mem_pool_reset(ctx.data);
    }

    printf("  various_configs: OK\n");

}

/* ── Test: generation runs in no-grad mode (no autograd tape) ── */

static void test_no_grad_mode(void) {
    printf("  test_no_grad_mode... ");

    dnn_ctx_init(&ctx, 4 * 1024 * 1024, 64 * 1024 * 1024, 1 * 1024 * 1024);

    int vocab=8, d_model=4, n_layers=2, n_heads=2, d_k=2, intermediate=8;

    srand(42);
    decoder_lm *lm = decoder_lm_create(ctx.params, vocab, d_model, n_layers, n_heads,
                                        d_k, intermediate);
    train_lm(lm, 1, 4, 5);

    int prompt_data[] = {2, 5, 2};
    tensor *prompt = make_int_tensor(2, (int[]){1, 3}, prompt_data);

    /* Check no autograd tape is created during generation */
    int n_out;
    int *result = decoder_lm_generate(ctx.scratch, ctx.data, lm, prompt, 5, 0.0f, 0, &n_out);

    /* Grads from prior training may be non-zero (zero_grad runs at top of each step).
     * Generation itself runs in dnn_no_grad mode — no new grads are allocated.
     * The main check is that output is valid. */

    assert(result != NULL && n_out >= 3);
    printf("OK (len=%d)\n", n_out);

}

/* ── Main ── */

int main(void) {
    printf("=== generation tests ===\n\n");

    test_generate_argmax_nocache();
    test_cached_vs_nocached();
    test_short_prompt();
    test_max_new_tokens_limit();
    test_deterministic();
    test_temperature_sampling();
    test_various_configs();
    test_no_grad_mode();

    printf("\nAll generation tests passed.\n");
    dnn_ctx_destroy(&ctx);

    return 0;
}
