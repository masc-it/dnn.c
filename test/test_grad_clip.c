#include "dnn.h"
#include "transformer.h"
#include "optim.h"
#include <stdio.h>
#include <assert.h>
#include <math.h>
#include <string.h>
#include <stdlib.h>

#define EPS 1e-5f

/* ── Helpers ── */

static float total_grad_norm(tensor **params, int n_params) {
    double sum_sq = 0.0;
    for (int i = 0; i < n_params; i++) {
        float *g = tensor_grad(params[i]);
        if (!g) continue;
        int n = tensor_numel(params[i]);
        for (int j = 0; j < n; j++)
            sum_sq += (double)g[j] * (double)g[j];
    }
    return sqrtf((float)sum_sq);
}

static float max_abs_grad(tensor **params, int n_params) {
    float mx = 0.0f;
    for (int i = 0; i < n_params; i++) {
        float *g = tensor_grad(params[i]);
        if (!g) continue;
        int n = tensor_numel(params[i]);
        for (int j = 0; j < n; j++) {
            float a = fabsf(g[j]);
            if (a > mx) mx = a;
        }
    }
    return mx;
}

/* ── Test 1: clip_grad_norm basic — large grads get scaled ── */

static void test_clip_norm_basic(void) {
    printf("  test_clip_norm_basic... ");
    mem_pool params  = mem_pool_create(4 * 1024 * 1024);
    mem_pool scratch = mem_pool_create(4 * 1024 * 1024);
    mem_pool data    = mem_pool_create(1 * 1024 * 1024);
    mem_pool_set_defaults(&params, &scratch, &data);

    srand(42);

    tensor *a = tensor_randn(1, (int[]){100}, 1);
    tensor *b = tensor_randn(1, (int[]){50}, 1);
    tensor *c = tensor_randn(1, (int[]){25}, 1);
    tensor *all[] = {a, b, c};
    int n = 3;

    /* Build graph: loss = sum(a*2) + sum(b) + sum(c) — separate params, each summed */
    tensor *scale = tensor_zeros(1, (int[]){1}, 0);
    tensor_data_ptr(scale)[0] = 2.0f; scale->contiguous = 1;
    tensor *loss = tensor_add(tensor_sum(tensor_mul(a, scale), -1), tensor_sum(b, -1));
    loss = tensor_add(loss, tensor_sum(c, -1));
    dnn_backward(loss);

    float norm_before = total_grad_norm(all, n);
    assert(norm_before > 0.0f && "grad norm should be > 0");

    float max_norm = 1.0f;
    float returned_norm = clip_grad_norm(all, n, max_norm);

    float norm_after = total_grad_norm(all, n);
    assert(returned_norm == norm_before && "return value should be norm BEFORE clipping");
    assert(norm_after <= max_norm * (1.0f + EPS) && "norm after clip should be <= max_norm");

    if (norm_before > max_norm) {
        assert(fabsf(norm_after - max_norm) < 0.01f && "norm after should be close to max_norm");
    }

    printf("OK (norm %.4f -> %.4f, max_norm=%.1f)\n",
           norm_before, norm_after, max_norm);
    mem_pool_destroy(&params);
    mem_pool_destroy(&scratch);
    mem_pool_destroy(&data);
}

/* ── Test 2: no clipping when norm < max_norm ── */

static void test_clip_norm_noop(void) {
    printf("  test_clip_norm_noop... ");
    mem_pool params  = mem_pool_create(4 * 1024 * 1024);
    mem_pool scratch = mem_pool_create(4 * 1024 * 1024);
    mem_pool data    = mem_pool_create(1 * 1024 * 1024);
    mem_pool_set_defaults(&params, &scratch, &data);

    srand(42);

    tensor *a = tensor_randn(1, (int[]){100}, 1);
    tensor *all[] = {a};
    int n = 1;

    tensor *scale = tensor_zeros(1, (int[]){1}, 0);
    tensor_data_ptr(scale)[0] = 0.001f; scale->contiguous = 1;
    tensor *loss = tensor_sum(tensor_mul(a, scale), -1);
    dnn_backward(loss);

    float norm_before = total_grad_norm(all, n);
    assert(norm_before > 0.0f);

    float max_norm = 1000.0f;
    float returned = clip_grad_norm(all, n, max_norm);

    float norm_after = total_grad_norm(all, n);
    assert(fabsf(returned - norm_before) < 1e-6f);
    assert(fabsf(norm_after - norm_before) < 1e-6f && "norm should be unchanged");

    printf("OK (norm %.6e unchanged)\n", norm_before);
    mem_pool_destroy(&params);
    mem_pool_destroy(&scratch);
    mem_pool_destroy(&data);
}

/* ── Test 3: max_norm <= 0 is no-op ── */

static void test_clip_norm_zero_noop(void) {
    printf("  test_clip_norm_zero_noop... ");
    mem_pool params  = mem_pool_create(4 * 1024 * 1024);
    mem_pool scratch = mem_pool_create(4 * 1024 * 1024);
    mem_pool data    = mem_pool_create(1 * 1024 * 1024);
    mem_pool_set_defaults(&params, &scratch, &data);

    srand(42);

    tensor *a = tensor_randn(1, (int[]){100}, 1);
    tensor *all[] = {a};
    int n = 1;

    tensor *two = tensor_zeros(1, (int[]){1}, 0);
    tensor_data_ptr(two)[0] = 2.0f; two->contiguous = 1;
    tensor *loss = tensor_sum(tensor_mul(a, two), -1);
    dnn_backward(loss);

    int n_el = tensor_numel(a);
    float *g_copy = malloc(n_el * sizeof(float));
    memcpy(g_copy, tensor_grad(a), n_el * sizeof(float));

    clip_grad_norm(all, n, 0.0f);

    float *g_after = tensor_grad(a);
    float max_diff = 0.0f;
    for (int i = 0; i < n_el; i++)
        if (fabsf(g_after[i] - g_copy[i]) > max_diff)
            max_diff = fabsf(g_after[i] - g_copy[i]);
    assert(max_diff < 1e-6f && "max_norm=0 should not change grads");

    clip_grad_norm(all, n, -1.0f);
    max_diff = 0.0f;
    for (int i = 0; i < n_el; i++)
        if (fabsf(g_after[i] - g_copy[i]) > max_diff)
            max_diff = fabsf(g_after[i] - g_copy[i]);
    assert(max_diff < 1e-6f && "max_norm<0 should not change grads");

    printf("OK (no-op)\n");
    free(g_copy);
    mem_pool_destroy(&params);
    mem_pool_destroy(&scratch);
    mem_pool_destroy(&data);
}

/* ── Test 4: extreme clip (very small max_norm) ── */

static void test_clip_norm_extreme(void) {
    printf("  test_clip_norm_extreme... ");
    mem_pool params  = mem_pool_create(4 * 1024 * 1024);
    mem_pool scratch = mem_pool_create(4 * 1024 * 1024);
    mem_pool data    = mem_pool_create(1 * 1024 * 1024);
    mem_pool_set_defaults(&params, &scratch, &data);

    srand(42);

    tensor *a = tensor_randn(1, (int[]){20}, 1);
    tensor *all[] = {a};
    int n = 1;

    tensor *hundred = tensor_zeros(1, (int[]){1}, 0);
    tensor_data_ptr(hundred)[0] = 100.0f; hundred->contiguous = 1;
    tensor *loss = tensor_sum(tensor_mul(a, hundred), -1);
    dnn_backward(loss);

    float norm_before = total_grad_norm(all, n);

    float max_norm = 0.01f;
    float returned = clip_grad_norm(all, n, max_norm);

    float norm_after = total_grad_norm(all, n);
    assert(returned == norm_before && "return = norm before clip");
    assert(norm_after <= max_norm + EPS && "after extreme clip, norm <= max_norm");
    assert(fabsf(norm_after - max_norm) < 0.001f && "should be close to max_norm");

    printf("OK (norm %.4e -> %.4e, max_norm=%.2e)\n",
           norm_before, norm_after, max_norm);
    mem_pool_destroy(&params);
    mem_pool_destroy(&scratch);
    mem_pool_destroy(&data);
}

/* ── Test 5: clip_grad_value basic ── */

static void test_clip_value_basic(void) {
    printf("  test_clip_value_basic... ");
    mem_pool params  = mem_pool_create(4 * 1024 * 1024);
    mem_pool scratch = mem_pool_create(4 * 1024 * 1024);
    mem_pool data    = mem_pool_create(1 * 1024 * 1024);
    mem_pool_set_defaults(&params, &scratch, &data);

    srand(42);

    tensor *a = tensor_randn(1, (int[]){100}, 1);
    tensor *b = tensor_randn(1, (int[]){50}, 1);
    tensor *all[] = {a, b};
    int n = 2;

    tensor *two = tensor_zeros(1, (int[]){1}, 0);
    tensor_data_ptr(two)[0] = 2.0f; two->contiguous = 1;
    tensor *loss = tensor_add(tensor_sum(tensor_mul(a, two), -1),
                              tensor_sum(b, -1));
    dnn_backward(loss);

    float max_before = max_abs_grad(all, n);
    assert(max_before > 0.0f);

    float clip_val = 0.1f;
    clip_grad_value(all, n, clip_val);

    float max_after = max_abs_grad(all, n);
    assert(max_after <= clip_val + EPS && "all grad values should be <= clip_val");

    printf("OK (max_abs %.4e -> %.4e)\n", max_before, max_after);
    mem_pool_destroy(&params);
    mem_pool_destroy(&scratch);
    mem_pool_destroy(&data);
}

/* ── Test 6: clip_grad_value no-op for small values ── */

static void test_clip_value_noop(void) {
    printf("  test_clip_value_noop... ");
    mem_pool params  = mem_pool_create(4 * 1024 * 1024);
    mem_pool scratch = mem_pool_create(4 * 1024 * 1024);
    mem_pool data    = mem_pool_create(1 * 1024 * 1024);
    mem_pool_set_defaults(&params, &scratch, &data);

    srand(42);

    tensor *a = tensor_randn(1, (int[]){100}, 1);
    tensor *all[] = {a};
    int n = 1;

    tensor *small = tensor_zeros(1, (int[]){1}, 0);
    tensor_data_ptr(small)[0] = 0.001f; small->contiguous = 1;
    tensor *loss = tensor_sum(tensor_mul(a, small), -1);
    dnn_backward(loss);

    int n_el = tensor_numel(a);
    float *g_copy = malloc(n_el * sizeof(float));
    memcpy(g_copy, tensor_grad(a), n_el * sizeof(float));

    clip_grad_value(all, n, 1000.0f);

    float max_diff = 0.0f;
    for (int i = 0; i < n_el; i++)
        if (fabsf(tensor_grad(a)[i] - g_copy[i]) > max_diff)
            max_diff = fabsf(tensor_grad(a)[i] - g_copy[i]);
    assert(max_diff < 1e-6f && "large clip_value should not change grads");

    printf("OK (no-op)\n");
    free(g_copy);
    mem_pool_destroy(&params);
    mem_pool_destroy(&scratch);
    mem_pool_destroy(&data);
}

/* ── Test 7: clip_grad_value <= 0 is no-op ── */

static void test_clip_value_zero_noop(void) {
    printf("  test_clip_value_zero_noop... ");
    mem_pool params  = mem_pool_create(4 * 1024 * 1024);
    mem_pool scratch = mem_pool_create(4 * 1024 * 1024);
    mem_pool data    = mem_pool_create(1 * 1024 * 1024);
    mem_pool_set_defaults(&params, &scratch, &data);

    srand(42);

    tensor *a = tensor_randn(1, (int[]){100}, 1);
    tensor *all[] = {a};
    int n = 1;

    tensor *two = tensor_zeros(1, (int[]){1}, 0);
    tensor_data_ptr(two)[0] = 2.0f; two->contiguous = 1;
    tensor *loss = tensor_sum(tensor_mul(a, two), -1);
    dnn_backward(loss);

    int n_el = tensor_numel(a);
    float *g_copy = malloc(n_el * sizeof(float));
    memcpy(g_copy, tensor_grad(a), n_el * sizeof(float));

    clip_grad_value(all, n, 0.0f);
    clip_grad_value(all, n, -0.5f);

    float max_diff = 0.0f;
    for (int i = 0; i < n_el; i++)
        if (fabsf(tensor_grad(a)[i] - g_copy[i]) > max_diff)
            max_diff = fabsf(tensor_grad(a)[i] - g_copy[i]);
    assert(max_diff < 1e-6f && "clip_value<=0 should not change grads");

    printf("OK (no-op)\n");
    free(g_copy);
    mem_pool_destroy(&params);
    mem_pool_destroy(&scratch);
    mem_pool_destroy(&data);
}

/* ── Test 8: clip in training step (via decoder_lm_train_step) ── */

static void test_clip_in_training_step(void) {
    printf("  test_clip_in_training_step... ");
    mem_pool params  = mem_pool_create(8 * 1024 * 1024);
    mem_pool scratch = mem_pool_create(64 * 1024 * 1024);
    mem_pool data    = mem_pool_create(1 * 1024 * 1024);
    mem_pool_set_defaults(&params, &scratch, &data);

    int B=2, N=4, vocab=8, d_model=4, n_layers=2, n_heads=2, d_k=2, intermediate=8;

    srand(42);
    decoder_lm *lm = decoder_lm_create(vocab, d_model, n_layers, n_heads,
                                        d_k, intermediate);

    tensor *all[256];
    int np = 0;
    all[np++] = lm->embedding_table; all[np++] = lm->norm_weight; all[np++] = lm->norm_bias;
    all[np++] = lm->lm_head->weight; all[np++] = lm->lm_head->bias;
    for (int i = 0; i < lm->n_layers; i++) {
        transformer_block *b = lm->blocks[i];
        all[np++] = b->q_proj->weight; all[np++] = b->q_proj->bias;
        all[np++] = b->k_proj->weight; all[np++] = b->k_proj->bias;
        all[np++] = b->v_proj->weight; all[np++] = b->v_proj->bias;
        all[np++] = b->out_proj->weight; all[np++] = b->out_proj->bias;
        all[np++] = b->attn_norm_weight; all[np++] = b->attn_norm_bias;
        all[np++] = b->ffn_norm_weight; all[np++] = b->ffn_norm_bias;
        all[np++] = b->ffn->gate_proj->weight; all[np++] = b->ffn->gate_proj->bias;
        all[np++] = b->ffn->up_proj->weight; all[np++] = b->ffn->up_proj->bias;
        all[np++] = b->ffn->down_proj->weight; all[np++] = b->ffn->down_proj->bias;
    }

    adamw_opt *opt = adamw_create(all, np, 0.001f,
                                   0.9f, 0.999f, 1e-8f, 0.01f);

    int ids[] = {3, 6, 7, 0, 4, 3, 4, 7};
    tensor *input_ids = tensor_zeros_data(2, (int[]){B, N});
    memcpy(input_ids->data, ids, B * N * sizeof(int));

    tensor *loss = decoder_lm_train_step(lm, input_ids, opt, 1.0f, NULL);
    float loss_val = tensor_data_ptr(loss)[0];
    assert(isfinite(loss_val) && "loss non-finite after clip training step");
    assert(loss_val > 0.0f && "loss should be positive");

    printf("OK (loss=%.6f, clip=1.0)\n", loss_val);
    mem_pool_destroy(&params);
    mem_pool_destroy(&scratch);
    mem_pool_destroy(&data);
}

/* ── Test 9: grad norm is reduced when using clip in multi-step training ── */

static void test_clip_reduces_norm_during_training(void) {
    printf("  test_clip_reduces_norm_during_training...\n");
    mem_pool params  = mem_pool_create(8 * 1024 * 1024);
    mem_pool scratch = mem_pool_create(64 * 1024 * 1024);
    mem_pool data    = mem_pool_create(1 * 1024 * 1024);
    mem_pool_set_defaults(&params, &scratch, &data);

    int B=2, N=4, vocab=8, d_model=4, n_layers=2, n_heads=2, d_k=2, intermediate=8;

    srand(42);
    decoder_lm *lm = decoder_lm_create(vocab, d_model, n_layers, n_heads,
                                        d_k, intermediate);
    tensor *all[256];
    int np = 0;
    all[np++] = lm->embedding_table; all[np++] = lm->norm_weight; all[np++] = lm->norm_bias;
    all[np++] = lm->lm_head->weight; all[np++] = lm->lm_head->bias;
    for (int i = 0; i < lm->n_layers; i++) {
        transformer_block *b = lm->blocks[i];
        all[np++] = b->q_proj->weight; all[np++] = b->q_proj->bias;
        all[np++] = b->k_proj->weight; all[np++] = b->k_proj->bias;
        all[np++] = b->v_proj->weight; all[np++] = b->v_proj->bias;
        all[np++] = b->out_proj->weight; all[np++] = b->out_proj->bias;
        all[np++] = b->attn_norm_weight; all[np++] = b->attn_norm_bias;
        all[np++] = b->ffn_norm_weight; all[np++] = b->ffn_norm_bias;
        all[np++] = b->ffn->gate_proj->weight; all[np++] = b->ffn->gate_proj->bias;
        all[np++] = b->ffn->up_proj->weight; all[np++] = b->ffn->up_proj->bias;
        all[np++] = b->ffn->down_proj->weight; all[np++] = b->ffn->down_proj->bias;
    }

    adamw_opt *opt = adamw_create(all, np, 0.001f,
                                   0.9f, 0.999f, 1e-8f, 0.01f);

    int ids[] = {3, 6, 7, 0, 4, 3, 4, 7};
    float clip_norm = 1.0f;

    for (int step = 0; step < 3; step++) {
        mem_pool_reset(&scratch);
        mem_pool_reset(&data);

        tensor *input_ids = tensor_zeros_data(2, (int[]){B, N});
        memcpy(input_ids->data, ids, B * N * sizeof(int));

        tensor *loss = decoder_lm_train_step(lm, input_ids, opt, clip_norm, NULL);
        float lv = tensor_data_ptr(loss)[0];
        assert(isfinite(lv));
        printf("    step %d: loss=%.6f (clip=%.1f)\n", step, lv, clip_norm);
    }

    printf("  clip_reduces_norm: OK\n");
    mem_pool_destroy(&params);
    mem_pool_destroy(&scratch);
    mem_pool_destroy(&data);
}

/* ── Test 10: clip_grad_norm returns correct value (norm before) ── */

static void test_clip_norm_return_value(void) {
    printf("  test_clip_norm_return_value... ");
    mem_pool params  = mem_pool_create(4 * 1024 * 1024);
    mem_pool scratch = mem_pool_create(4 * 1024 * 1024);
    mem_pool data    = mem_pool_create(1 * 1024 * 1024);
    mem_pool_set_defaults(&params, &scratch, &data);

    srand(42);

    tensor *a = tensor_randn(1, (int[]){20}, 1);
    tensor *all[] = {a};
    int n = 1;

    tensor *two = tensor_zeros(1, (int[]){1}, 0);
    tensor_data_ptr(two)[0] = 2.0f; two->contiguous = 1;
    tensor *loss = tensor_sum(tensor_mul(a, two), -1);
    dnn_backward(loss);

    float norm_before = total_grad_norm(all, n);
    float returned = clip_grad_norm(all, n, 0.5f);

    assert(returned == norm_before && "return value must be norm before clipping");

    float norm_after = total_grad_norm(all, n);
    assert(norm_after <= 0.5f + EPS);

    printf("OK (returned=%.6f, before=%.6f, after=%.6f)\n",
           returned, norm_before, norm_after);
    mem_pool_destroy(&params);
    mem_pool_destroy(&scratch);
    mem_pool_destroy(&data);
}

/* ── Main ── */

int main(void) {
    printf("=== gradient clipping tests ===\n\n");

    test_clip_norm_basic();
    test_clip_norm_noop();
    test_clip_norm_zero_noop();
    test_clip_norm_extreme();
    test_clip_norm_return_value();
    test_clip_value_basic();
    test_clip_value_noop();
    test_clip_value_zero_noop();
    test_clip_in_training_step();
    test_clip_reduces_norm_during_training();

    printf("\nAll gradient clipping tests passed.\n");
    return 0;
}
