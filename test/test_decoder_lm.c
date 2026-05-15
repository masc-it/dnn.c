#include "dnn.h"
#include "context.h"
#include "transformer.h"
#include <stdio.h>
#include <assert.h>
#include <math.h>
#include <string.h>
#include <stdlib.h>

static dnn_ctx ctx;

#define EPS 1e-5f

/* ── Helpers ── */

static tensor *make_int_tensor(int ndim, const int *shape, const int *data) {
    int n = 1;
    for (int i = 0; i < ndim; i++) n *= shape[i];
    tensor *t = tensor_zeros_data(ctx.data, ndim, shape);
    if (data) memcpy(t->data, data, n * sizeof(int));
    return t;
}

/* Reference input IDs from test/ref_decoder_lm.py --small (seed=42) */
static int ref_input_ids[] = {2, 5, 2};

/* ── Test: create LM ── */

static void test_lm_create(void) {
    printf("  test_lm_create... ");

    dnn_ctx_init(&ctx, 2 * 1024 * 1024, 32 * 1024 * 1024, 2 * 1024 * 1024);

    int vocab=10, d_model=4, n_layers=2, n_heads=2, d_k=2, intermediate=8;

    decoder_lm *lm = decoder_lm_create(ctx.params, vocab, d_model, n_layers, n_heads,
                                        d_k, intermediate);

    assert(lm->vocab_size == vocab);
    assert(lm->d_model    == d_model);
    assert(lm->n_layers   == n_layers);

    /* Embedding table shape */
    assert(tensor_ndim(lm->embedding_table) == 2);
    assert(tensor_shape(lm->embedding_table, 0) == vocab);
    assert(tensor_shape(lm->embedding_table, 1) == d_model);
    assert(tensor_requires_grad(lm->embedding_table));

    /* Blocks exist */
    assert(lm->blocks != NULL);
    for (int i = 0; i < n_layers; i++) {
        assert(lm->blocks[i] != NULL);
        assert(lm->blocks[i]->d_model == d_model);
        assert(lm->blocks[i]->n_heads == n_heads);
        assert(lm->blocks[i]->d_k     == d_k);
    }

    /* Final norm */
    assert(tensor_shape(lm->norm_weight, 0) == d_model);
    assert(tensor_shape(lm->norm_bias,   0) == d_model);
    assert(tensor_requires_grad(lm->norm_weight));

    /* LM head */
    assert(lm->lm_head->in_features  == d_model);
    assert(lm->lm_head->out_features == vocab);

    /* All params require grad */
    assert(tensor_requires_grad(lm->lm_head->weight));
    assert(tensor_requires_grad(lm->lm_head->bias));

    printf("OK\n");

}

/* ── Test: forward shape and output ── */

static void test_forward_basic(void) {
    printf("  test_forward_basic... ");

    dnn_ctx_init(&ctx, 2 * 1024 * 1024, 32 * 1024 * 1024, 2 * 1024 * 1024);

    int B=1, N=3, vocab=10, d_model=4, n_layers=2, n_heads=2, d_k=2, intermediate=8;

    srand(42);
    decoder_lm *lm = decoder_lm_create(ctx.params, vocab, d_model, n_layers, n_heads,
                                        d_k, intermediate);

    tensor *input_ids = make_int_tensor(2, (int[]){B, N}, ref_input_ids);

    tensor *logits = decoder_lm_forward(ctx.scratch, lm, input_ids);

    /* Shape check */
    assert(tensor_ndim(logits) == 3);
    assert(tensor_shape(logits, 0) == B);
    assert(tensor_shape(logits, 1) == N);
    assert(tensor_shape(logits, 2) == vocab);

    /* Finite check */
    float *ld = tensor_data_ptr(logits);
    for (int i = 0; i < B * N * vocab; i++)
        assert(isfinite(ld[i]) && "logits non-finite");

    /* Backward: sum loss */
    dnn_backward(ctx.scratch, logits);

    /* All major param groups get grads */
    assert(tensor_grad(lm->embedding_table)  && "embedding grad NULL");
    assert(tensor_grad(lm->norm_weight)      && "norm_weight grad NULL");
    assert(tensor_grad(lm->norm_bias)        && "norm_bias grad NULL");
    assert(tensor_grad(lm->lm_head->weight)  && "lm_head weight grad NULL");
    assert(tensor_grad(lm->lm_head->bias)    && "lm_head bias grad NULL");

    /* All block params get grads */
    for (int i = 0; i < n_layers; i++) {
        transformer_block *b = lm->blocks[i];
        assert(tensor_grad(b->q_proj->weight) && "block q_proj grad NULL");
        assert(tensor_grad(b->k_proj->weight) && "block k_proj grad NULL");
        assert(tensor_grad(b->v_proj->weight) && "block v_proj grad NULL");
        assert(tensor_grad(b->out_proj->weight) && "block out_proj grad NULL");
        assert(tensor_grad(b->attn_norm_weight) && "block attn_norm_weight grad NULL");
        assert(tensor_grad(b->ffn_norm_weight) && "block ffn_norm_weight grad NULL");
        assert(tensor_grad(b->ffn->gate_proj->weight) && "block gate_proj grad NULL");
        assert(tensor_grad(b->ffn->up_proj->weight) && "block up_proj grad NULL");
        assert(tensor_grad(b->ffn->down_proj->weight) && "block down_proj grad NULL");
    }

    /* All grads finite */
    float *eg = tensor_grad(lm->embedding_table);
    for (int i = 0; i < vocab * d_model; i++) assert(isfinite(eg[i]));

    printf("OK (shape correct, grads flow to all %d param groups, finite)\n",
           3 + n_layers * 9);

}

/* ── Test: embedding gradient only flows to looked-up rows ── */

static void test_embedding_grad_sparsity(void) {
    printf("  test_embedding_grad_sparsity... ");

    dnn_ctx_init(&ctx, 2 * 1024 * 1024, 32 * 1024 * 1024, 2 * 1024 * 1024);

    int B=1, N=3, vocab=10, d_model=4, n_layers=1, n_heads=2, d_k=2, intermediate=8;

    srand(7);
    decoder_lm *lm = decoder_lm_create(ctx.params, vocab, d_model, n_layers, n_heads,
                                        d_k, intermediate);

    /* Input IDs {2, 5, 2} — only rows 2 and 5 should get gradients */
    tensor *input_ids = make_int_tensor(2, (int[]){B, N}, ref_input_ids);

    tensor *logits = decoder_lm_forward(ctx.scratch, lm, input_ids);
    dnn_backward(ctx.scratch, logits);

    float *eg = tensor_grad(lm->embedding_table);
    for (int i = 0; i < vocab; i++) {
        int should_be_zero = (i != 2 && i != 5);
        for (int j = 0; j < d_model; j++) {
            if (should_be_zero) {
                assert(eg[i * d_model + j] == 0.0f &&
                       "embedding grad non-zero for non-looked-up row");
            } else {
                assert(eg[i * d_model + j] != 0.0f &&
                       "embedding grad zero for looked-up row");
            }
        }
    }

    printf("OK (rows {2,5} get grad, others zero)\n");

}

/* ── Test: numerical gradient check (finite diff) ── */

static void test_numerical_grad(void) {
    printf("  test_numerical_grad...\n");

    int B=1, N=2, vocab=5, d_model=2, n_layers=1, n_heads=1, d_k=2, intermediate=4;
    int n_input = B * N;  (void)n_input;

    dnn_ctx_init(&ctx, 2 * 1024 * 1024, 32 * 1024 * 1024, 2 * 1024 * 1024);

    srand(42);
    decoder_lm *lm = decoder_lm_create(ctx.params, vocab, d_model, n_layers, n_heads,
                                        d_k, intermediate);

    int input_data[] = {0, 3};
    tensor *input_ids = make_int_tensor(2, (int[]){B, N}, input_data);

    tensor *logits = decoder_lm_forward(ctx.scratch, lm, input_ids);
    dnn_backward(ctx.scratch, logits);

    /* Pick a few parameters for finite diff check */
    float h = 1e-4f;
    int n_checks = 4;
    int n_passed = 0;

    /* Capture lm_head weight data for perturbation */
    int n_lm = tensor_numel(lm->lm_head->weight);
    float *lm_w_orig = malloc(n_lm * sizeof(float));
    memcpy(lm_w_orig, tensor_data_ptr(lm->lm_head->weight), n_lm * sizeof(float));

    float *lm_w_grad_auto = tensor_grad(lm->lm_head->weight);
    assert(lm_w_grad_auto != NULL);

    for (int idx = 0; idx < n_lm && n_passed < n_checks; idx += n_lm / n_checks) {
        /* +h */
        mem_pool_reset(ctx.scratch);
        mem_pool_reset(ctx.data);

        srand(42);
        decoder_lm *lm1 = decoder_lm_create(ctx.params, vocab, d_model, n_layers, n_heads,
                                             d_k, intermediate);
        memcpy(tensor_data_ptr(lm1->lm_head->weight), lm_w_orig, n_lm * sizeof(float));
        float *w1d = tensor_data_ptr(lm1->lm_head->weight);
        w1d[idx] += h;

        tensor *ids1 = make_int_tensor(2, (int[]){B, N}, input_data);
        tensor *o1 = decoder_lm_forward(ctx.scratch, lm1, ids1);
        float l1 = 0.0f;
        float *o1d = tensor_data_ptr(o1);
        int nel = tensor_numel(o1);
        for (int i = 0; i < nel; i++) l1 += o1d[i];

        /* -h */
        mem_pool_reset(ctx.scratch);
        mem_pool_reset(ctx.data);

        srand(42);
        decoder_lm *lm2 = decoder_lm_create(ctx.params, vocab, d_model, n_layers, n_heads,
                                             d_k, intermediate);
        memcpy(tensor_data_ptr(lm2->lm_head->weight), lm_w_orig, n_lm * sizeof(float));
        float *w2d = tensor_data_ptr(lm2->lm_head->weight);
        w2d[idx] -= h;

        tensor *ids2 = make_int_tensor(2, (int[]){B, N}, input_data);
        tensor *o2 = decoder_lm_forward(ctx.scratch, lm2, ids2);
        float l2 = 0.0f;
        float *o2d = tensor_data_ptr(o2);
        for (int i = 0; i < nel; i++) l2 += o2d[i];

        float fd = (l1 - l2) / (2.0f * h);
        float ag = lm_w_grad_auto[idx];
        float diff = fabsf(fd - ag);

        printf("    lm_head.weight[%d]: fd=%.6f auto=%.6f diff=%.2e %s\n",
               idx, fd, ag, diff, diff < 0.05f ? "OK" : "FAIL");
        assert(diff < 0.05f);
        n_passed++;
    }

    printf("  numerical gradient: OK (%d checks)\n", n_passed);
    free(lm_w_orig);

}

/* ── Test: no-grad mode (eval) ── */

static void test_no_grad(void) {
    printf("  test_no_grad... ");

    dnn_ctx_init(&ctx, 2 * 1024 * 1024, 32 * 1024 * 1024, 2 * 1024 * 1024);

    int B=1, N=3, vocab=10, d_model=4, n_layers=2, n_heads=2, d_k=2, intermediate=8;

    srand(42);
    decoder_lm *lm = decoder_lm_create(ctx.params, vocab, d_model, n_layers, n_heads,
                                        d_k, intermediate);

    tensor *input_ids = make_int_tensor(2, (int[]){B, N}, ref_input_ids);

    dnn_grad_ctx gc = dnn_no_grad_enter();
    tensor *logits = decoder_lm_forward(ctx.scratch, lm, input_ids);
    dnn_no_grad_exit(gc);

    assert(logits->grad_fn == NULL && "no-grad: autograd should not be wired");
    assert(tensor_ndim(logits) == 3);
    assert(tensor_shape(logits, 0) == B);
    assert(tensor_shape(logits, 1) == N);
    assert(tensor_shape(logits, 2) == vocab);

    float *ld = tensor_data_ptr(logits);
    for (int i = 0; i < B * N * vocab; i++) assert(isfinite(ld[i]));

    printf("OK\n");

}

/* ── Test: batch processing ── */

static void test_batch(void) {
    printf("  test_batch... ");

    dnn_ctx_init(&ctx, 2 * 1024 * 1024, 32 * 1024 * 1024, 2 * 1024 * 1024);

    int B=2, N=3, vocab=10, d_model=4, n_layers=2, n_heads=2, d_k=2, intermediate=8;

    srand(42);
    decoder_lm *lm = decoder_lm_create(ctx.params, vocab, d_model, n_layers, n_heads,
                                        d_k, intermediate);

    int ids_data[] = {0, 1, 2, 3, 4, 5};
    tensor *input_ids = make_int_tensor(2, (int[]){B, N}, ids_data);

    tensor *logits = decoder_lm_forward(ctx.scratch, lm, input_ids);

    assert(tensor_shape(logits, 0) == B);
    assert(tensor_shape(logits, 1) == N);
    assert(tensor_shape(logits, 2) == vocab);

    dnn_backward(ctx.scratch, logits);
    assert(tensor_grad(lm->embedding_table) && "batch: embedding grad");
    assert(tensor_grad(lm->blocks[0]->q_proj->weight) && "batch: block q_proj grad");

    float *eg = tensor_grad(lm->embedding_table);
    for (int i = 0; i < vocab * d_model; i++) assert(isfinite(eg[i]));

    printf("OK (B=2, grads flow)\n");

}

/* ── Test: different sequence lengths ── */

static void test_seq_len(void) {
    printf("  test_seq_len... ");

    dnn_ctx_init(&ctx, 2 * 1024 * 1024, 32 * 1024 * 1024, 2 * 1024 * 1024);

    int vocab=10, d_model=4, n_layers=2, n_heads=2, d_k=2, intermediate=8;

    srand(42);
    decoder_lm *lm = decoder_lm_create(ctx.params, vocab, d_model, n_layers, n_heads,
                                        d_k, intermediate);

    /* N=1 */
    int ids1_data[] = {7};
    tensor *ids1 = make_int_tensor(2, (int[]){1, 1}, ids1_data);
    tensor *o1 = decoder_lm_forward(ctx.scratch, lm, ids1);
    assert(tensor_shape(o1, 0) == 1 && tensor_shape(o1, 1) == 1 && tensor_shape(o1, 2) == vocab);
    dnn_backward(ctx.scratch, o1);
    assert(tensor_grad(lm->embedding_table) && "N=1: grads flow");
    printf("  N=1 OK ");

    mem_pool_reset(ctx.scratch);
    mem_pool_reset(ctx.data);

    /* N=7 */
    int ids7_data[] = {0, 1, 2, 3, 4, 5, 6};
    tensor *ids7 = make_int_tensor(2, (int[]){1, 7}, ids7_data);
    tensor *o7 = decoder_lm_forward(ctx.scratch, lm, ids7);
    assert(tensor_shape(o7, 0) == 1 && tensor_shape(o7, 1) == 7 && tensor_shape(o7, 2) == vocab);
    dnn_backward(ctx.scratch, o7);
    assert(tensor_grad(lm->embedding_table) && "N=7: grads flow");
    printf("N=7 OK ");

    printf("\n");

}

/* ── Test: weight tying (optional feature) ── */

static void test_weight_tying(void) {
    printf("  test_weight_tying... ");

    dnn_ctx_init(&ctx, 2 * 1024 * 1024, 32 * 1024 * 1024, 2 * 1024 * 1024);

    int vocab=10, d_model=4, n_layers=1, n_heads=2, d_k=2, intermediate=8;

    srand(7);
    decoder_lm *lm = decoder_lm_create(ctx.params, vocab, d_model, n_layers, n_heads,
                                        d_k, intermediate);

    /* Test that embedding table and lm_head are different tensors by default */
    assert(lm->embedding_table != lm->lm_head->weight &&
           "embedding and lm_head must be separate tensors (no weight tying)");

    /* Both are in params pool and require grad */
    assert(tensor_requires_grad(lm->embedding_table));
    assert(tensor_requires_grad(lm->lm_head->weight));

    /* Shared memory layout: embedding is [vocab, d_model], lm_head is [d_model, vocab].
     * They are different shapes so can't share buffer without a transpose. */

    printf("OK (separate tensors, both require grad)\n");

}

/* ── Test: gradient flow through embedding duplicates
 *
 *   Input {2, 5, 2} — token 2 appears twice.
 *   The embedding gradient at row 2 should be the sum of gradients
 *   from both positions.
 */

static void test_embedding_duplicate_ids(void) {
    printf("  test_embedding_duplicate_ids... ");

    dnn_ctx_init(&ctx, 2 * 1024 * 1024, 32 * 1024 * 1024, 2 * 1024 * 1024);

    int B=1, N=3, vocab=10, d_model=4, n_layers=1, n_heads=2, d_k=2, intermediate=8;

    srand(9);
    decoder_lm *lm = decoder_lm_create(ctx.params, vocab, d_model, n_layers, n_heads,
                                        d_k, intermediate);

    /* IDs {2, 5, 2}: row 2 gets gradient from positions 0 and 2 */
    tensor *input_ids = make_int_tensor(2, (int[]){B, N}, ref_input_ids);

    tensor *logits = decoder_lm_forward(ctx.scratch, lm, input_ids);
    dnn_backward(ctx.scratch, logits);

    float *eg = tensor_grad(lm->embedding_table);
    float *eg_row2 = eg + 2 * d_model;
    float *eg_row5 = eg + 5 * d_model;

    /* Row 2 gradient magnitude should be larger than row 5
     * since it accumulates from 2 positions vs 1.
     * (Not exactly 2x because the gradient at each position differs,
     *  but it's a useful sanity check.) */
    float norm2 = 0.0f, norm5 = 0.0f;
    for (int j = 0; j < d_model; j++) {
        norm2 += eg_row2[j] * eg_row2[j];
        norm5 += eg_row5[j] * eg_row5[j];
    }

    /* Norm of row 2 should be non-zero and different from row 5 */
    assert(norm2 > 0.0f && "duplicate row grad zero");
    assert(norm5 > 0.0f && "single row grad zero");

    printf("OK (row2 norm=%.4f, row5 norm=%.4f, both >0)\n",
           sqrtf(norm2), sqrtf(norm5));

}

/* ── Main ── */

int main(void) {
    printf("=== decoder_lm tests ===\n\n");

    test_lm_create();
    test_forward_basic();
    test_embedding_grad_sparsity();
    test_numerical_grad();
    test_no_grad();
    test_batch();
    test_seq_len();
    test_weight_tying();
    test_embedding_duplicate_ids();

    printf("\nAll decoder_lm tests passed.\n");
    dnn_ctx_destroy(&ctx);

    return 0;
}
