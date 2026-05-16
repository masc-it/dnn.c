#include "dnn.h"
#include "gpt.h"
#include "vlm.h"
#include "context.h"
#include "optim.h"
#include <stdio.h>
#include <assert.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>

static dnn_ctx ctx;

static tensor *make_int_tensor(int ndim, const int *shape, const int *data) {
    int n = 1;
    for (int i = 0; i < ndim; i++) n *= shape[i];
    tensor *t = tensor_zeros_data(ctx.data, ndim, shape);
    if (data) memcpy(t->data, data, n * sizeof(int));
    return t;
}

/* ══════════════════════════════════════════════════════════════════
 *  Test 1: create tiny VLM and check shapes/params
 * ══════════════════════════════════════════════════════════════════ */

static void test_create_and_shapes(void) {
    printf("  test_create_and_shapes... ");
    dnn_ctx_init(&ctx, 8*1024*1024, 64*1024*1024, 1*1024*1024);
    srand(42);
    vision_lm *vlm = vision_lm_create(ctx.params, 16, 8, 1, 2, 4, 16,
                                       3, 8, 8, 4, 1);
    assert(vlm->n_img_tokens == 4);
    assert(vlm->patch_h == 2 && vlm->patch_w == 2);
    assert(vlm->d_model == 8 && vlm->vocab_size == 16);
    assert(vlm->image_pos != NULL);

    /* Check image_pos is in param list */
    int n_params;
    tensor **params = module_parameters(&vlm->base, &n_params);
    int found_ip = 0, found_pw = 0;
    for (int i = 0; i < n_params; i++) {
        if (params[i] == vlm->image_pos) found_ip = 1;
        if (params[i] == vlm->patch_embed->weight) found_pw = 1;
    }
    assert(found_ip && "image_pos must be registered");
    assert(found_pw && "patch_embed.weight must be registered");
    printf("OK (n_params=%d)\n", n_params);
}

/* ══════════════════════════════════════════════════════════════════
 *  Test 2: image_embeds shape
 * ══════════════════════════════════════════════════════════════════ */

static void test_image_embeds_shape(void) {
    printf("  test_image_embeds_shape... ");
    dnn_ctx_init(&ctx, 8*1024*1024, 64*1024*1024, 1*1024*1024);
    srand(42);
    vision_lm *vlm = vision_lm_create(ctx.params, 16, 8, 1, 2, 4, 16,
                                       3, 8, 8, 4, 1);
    tensor *img = tensor_randn(ctx.scratch, 4, (int[]){2, 3, 8, 8}, 0);
    tensor *e = vision_lm_image_embeds(ctx.scratch, vlm, img);
    assert(e->ndim == 3 && e->shape[0]==2 && e->shape[1]==4 && e->shape[2]==8);
    printf("OK (%dx%dx%d)\n", e->shape[0], e->shape[1], e->shape[2]);
}

/* ══════════════════════════════════════════════════════════════════
 *  Test 3: build_embeds shape
 * ══════════════════════════════════════════════════════════════════ */

static void test_build_embeds_shape(void) {
    printf("  test_build_embeds_shape... ");
    dnn_ctx_init(&ctx, 8*1024*1024, 64*1024*1024, 1*1024*1024);
    srand(42);
    vision_lm *vlm = vision_lm_create(ctx.params, 16, 8, 1, 2, 4, 16,
                                       3, 8, 8, 4, 0);
    tensor *img = tensor_randn(ctx.scratch, 4, (int[]){2, 3, 8, 8}, 0);
    int ids[10] = {0,1,2,3,4,5,6,7,8,9};
    tensor *txt = make_int_tensor(2, (int[]){2,5}, ids);
    tensor *e = vision_lm_build_embeds(ctx.scratch, vlm, img, txt);
    assert(e->ndim == 3 && e->shape[0]==2 && e->shape[1]==9 && e->shape[2]==8);
    printf("OK (%dx%dx%d)\n", e->shape[0], e->shape[1], e->shape[2]);
}

/* ══════════════════════════════════════════════════════════════════
 *  Test 4: forward shape
 * ══════════════════════════════════════════════════════════════════ */

static void test_forward_shape(void) {
    printf("  test_forward_shape... ");
    dnn_ctx_init(&ctx, 8*1024*1024, 128*1024*1024, 1*1024*1024);
    srand(42);
    vision_lm *vlm = vision_lm_create(ctx.params, 16, 8, 1, 2, 4, 16,
                                       3, 8, 8, 4, 0);
    tensor *img = tensor_randn(ctx.scratch, 4, (int[]){2, 3, 8, 8}, 0);
    int ids[10] = {0,1,2,3,4,5,6,7,8,9};
    tensor *txt = make_int_tensor(2, (int[]){2,5}, ids);
    tensor *logits = vision_lm_forward(ctx.scratch, vlm, img, txt);
    assert(logits->ndim==3 && logits->shape[0]==2 && logits->shape[1]==9 && logits->shape[2]==16);
    printf("OK (%dx%dx%d)\n", logits->shape[0], logits->shape[1], logits->shape[2]);
}

/* ══════════════════════════════════════════════════════════════════
 *  Test 5: forward_text_logits shape
 * ══════════════════════════════════════════════════════════════════ */

static void test_forward_text_logits_shape(void) {
    printf("  test_forward_text_logits_shape... ");
    dnn_ctx_init(&ctx, 8*1024*1024, 128*1024*1024, 1*1024*1024);
    srand(42);
    vision_lm *vlm = vision_lm_create(ctx.params, 16, 8, 1, 2, 4, 16,
                                       3, 8, 8, 4, 0);
    tensor *img = tensor_randn(ctx.scratch, 4, (int[]){2, 3, 8, 8}, 0);
    int ids[10] = {0,1,2,3,4,5,6,7,8,9};
    tensor *txt = make_int_tensor(2, (int[]){2,5}, ids);
    tensor *lt = vision_lm_forward_text_logits(ctx.scratch, vlm, img, txt);
    assert(lt->ndim==3 && lt->shape[0]==2 && lt->shape[1]==5 && lt->shape[2]==16);
    printf("OK (%dx%dx%d)\n", lt->shape[0], lt->shape[1], lt->shape[2]);
}

/* ══════════════════════════════════════════════════════════════════
 *  Test 6: train step returns finite loss
 * ══════════════════════════════════════════════════════════════════ */

static void test_train_step(void) {
    printf("  test_train_step... ");
    dnn_ctx_init(&ctx, 8*1024*1024, 128*1024*1024, 1*1024*1024);
    srand(42);
    vision_lm *vlm = vision_lm_create(ctx.params, 16, 8, 1, 2, 4, 16,
                                       3, 8, 8, 4, 1);
    int n_params;
    tensor **all_p = module_parameters(&vlm->base, &n_params);
    adamw_opt *opt = adamw_create(ctx.params, all_p, n_params, 0.001f,
                                   0.9f, 0.999f, 1e-8f, 0.01f);
    tensor *img = tensor_randn(ctx.scratch, 4, (int[]){2, 3, 8, 8}, 0);
    int ids[8] = {0,3,7,5, 1,4,7,2};
    int tgt[8] = {3,7,5,8, 4,7,2,9};
    tensor *inp = make_int_tensor(2, (int[]){2,4}, ids);
    tensor *tar = make_int_tensor(2, (int[]){2,4}, tgt);
    tensor *loss = vision_lm_train_step(ctx.scratch, vlm, img, inp, tar, opt, 0.0f, NULL);
    float lv = tensor_data_ptr(loss)[0];
    assert(isfinite(lv) && lv > 0.0f);
    printf("OK (loss=%.6f)\n", lv);
}

/* ══════════════════════════════════════════════════════════════════
 *  Test 7: gradients flow to all param groups
 * ══════════════════════════════════════════════════════════════════ */

static void test_gradients_flow(void) {
    printf("  test_gradients_flow...\n");
    dnn_ctx_init(&ctx, 8*1024*1024, 128*1024*1024, 1*1024*1024);
    srand(42);
    vision_lm *vlm = vision_lm_create(ctx.params, 16, 8, 1, 2, 4, 16,
                                       3, 8, 8, 4, 1);
    int n_params;
    tensor **all_p = module_parameters(&vlm->base, &n_params);
    adamw_opt *opt = adamw_create(ctx.params, all_p, n_params, 0.001f,
                                   0.9f, 0.999f, 1e-8f, 0.01f);
    assert(tensor_grad(vlm->patch_embed->weight) == NULL);
    assert(tensor_grad(vlm->image_pos) == NULL);
    tensor *img = tensor_randn(ctx.scratch, 4, (int[]){2, 3, 8, 8}, 0);
    int ids[8] = {0,3,7,5, 1,4,7,2};
    int tgt[8] = {3,7,5,8, 4,7,2,9};
    tensor *inp = make_int_tensor(2, (int[]){2,4}, ids);
    tensor *tar = make_int_tensor(2, (int[]){2,4}, tgt);
    tensor *loss = vision_lm_train_step(ctx.scratch, vlm, img, inp, tar, opt, 0.0f, NULL);
    float lv = tensor_data_ptr(loss)[0];
    printf("    loss=%.6f\n", lv);
    assert(tensor_grad(vlm->patch_embed->weight) != NULL);
    assert(tensor_grad(vlm->patch_embed->bias) != NULL);
    assert(tensor_grad(vlm->image_pos) != NULL);
    assert(tensor_grad(vlm->lm->embed->weight) != NULL);
    assert(tensor_grad(vlm->lm->norm->weight) != NULL);
    /* Check finite */
    float *ig = tensor_grad(vlm->image_pos);
    for (int i = 0; i < tensor_numel(vlm->image_pos); i++)
        assert(isfinite(ig[i]));
    float *wg = tensor_grad(vlm->patch_embed->weight);
    for (int i = 0; i < tensor_numel(vlm->patch_embed->weight); i++)
        assert(isfinite(wg[i]));
    printf("    OK\n");
}

/* ══════════════════════════════════════════════════════════════════
 *  Test 8: param count matches manual count
 * ══════════════════════════════════════════════════════════════════ */

static void test_param_count(void) {
    printf("  test_param_count... ");
    dnn_ctx_init(&ctx, 8*1024*1024, 64*1024*1024, 1*1024*1024);
    srand(42);
    vision_lm *vlm = vision_lm_create(ctx.params, 16, 8, 1, 2, 4, 16,
                                       3, 8, 8, 4, 1);
    long long total = vision_lm_num_parameters(vlm);
    /* Manual: patch=392 + image_pos=32 + embed=128 + norm=8 + lm_head=16 + block=744 = 1320 */
    assert(total == 1320);
    printf("OK (%lld)\n", total);
}

/* ══════════════════════════════════════════════════════════════════
 *  Test 9: image_pos disabled when use_image_pos=0
 * ══════════════════════════════════════════════════════════════════ */

static void test_image_pos_disabled(void) {
    printf("  test_image_pos_disabled... ");
    dnn_ctx_init(&ctx, 8*1024*1024, 64*1024*1024, 1*1024*1024);
    srand(42);
    vision_lm *vlm = vision_lm_create(ctx.params, 16, 8, 1, 2, 4, 16,
                                       3, 8, 8, 4, 0);
    assert(vlm->image_pos == NULL);
    assert(vlm->use_image_pos == 0);
    printf("OK\n");
}

/* ══════════════════════════════════════════════════════════════════
 *  Test 10: prefix-LM attention permits image-to-later-image
 * ══════════════════════════════════════════════════════════════════ */

static void test_prefix_lm_attention(void) {
    printf("  test_prefix_lm_attention... ");
    dnn_ctx_init(&ctx, 8*1024*1024, 64*1024*1024, 1*1024*1024);
    int B=1, H=1, N=6, d=4, I=3;
    srand(123);
    tensor *q = tensor_randn(ctx.scratch, 4, (int[]){B,H,N,d}, 0);
    tensor *k = tensor_randn(ctx.scratch, 4, (int[]){B,H,N,d}, 0);
    tensor *v = tensor_randn(ctx.scratch, 4, (int[]){B,H,N,d}, 0);
    tensor *oc = tensor_attention_ex(ctx.scratch, q,k,v,NULL, ATTENTION_CAUSAL, 0, NULL);
    tensor *op = tensor_attention_ex(ctx.scratch, q,k,v,NULL, ATTENTION_PREFIX_LM, I, NULL);
    float *ocd = tensor_data_ptr(oc), *opd = tensor_data_ptr(op);
    float diff_img = 0, diff_txt = 0;
    for (int j=0; j<d; j++) diff_img += fabsf(ocd[0*d+j] - opd[0*d+j]);
    for (int j=0; j<d; j++) diff_txt += fabsf(ocd[3*d+j] - opd[3*d+j]);
    assert(diff_img > 0 && "image row differs");
    assert(diff_txt < 1e-6f && "text row same");
    printf("OK (img_diff=%.6f txt_diff=%.6f)\n", diff_img, diff_txt);
}

/* ══════════════════════════════════════════════════════════════════
 *  Test 11: generation no-cache returns valid tokens
 * ══════════════════════════════════════════════════════════════════ */

static void test_generate_nocache(void) {
    printf("  test_generate_nocache... ");
    dnn_ctx_init(&ctx, 8*1024*1024, 128*1024*1024, 4*1024*1024);
    srand(42);
    vision_lm *vlm = vision_lm_create(ctx.params, 16, 8, 1, 2, 4, 16,
                                       3, 8, 8, 4, 0);
    vision_lm_init_weights(vlm);
    tensor *img_s = tensor_randn(ctx.scratch, 4, (int[]){1,3,8,8}, 0);
    tensor *img = tensor_zeros_data(ctx.data, 4, (int[]){1,3,8,8});
    memcpy(tensor_data_ptr(img), tensor_data_ptr(img_s), (size_t)(3*8*8)*sizeof(float));
    int prompt[2] = {0,1};
    tensor *pr = make_int_tensor(2, (int[]){1,2}, prompt);
    int n_out;
    int *res = vision_lm_generate(ctx.scratch, ctx.data, vlm, img, pr, 10, 0.0f, 0, &n_out);
    assert(res != NULL && n_out >= 2 && n_out <= 12);
    assert(res[0]==0 && res[1]==1);
    for (int i=0; i<n_out; i++) assert(res[i]>=0 && res[i]<16);
    printf("OK (n_out=%d)\n", n_out);
}

/* ══════════════════════════════════════════════════════════════════
 *  Test 12: cached generation matches no-cache (greedy)
 * ══════════════════════════════════════════════════════════════════ */

static void test_generate_cached_matches_nocache(void) {
    printf("  test_generate_cached_matches_nocache...\n");
    dnn_ctx_init(&ctx, 32*1024*1024, 512*1024*1024, 16*1024*1024);
    srand(42);
    vision_lm *vlm = vision_lm_create(ctx.params, 16, 8, 1, 2, 4, 16,
                                       3, 8, 8, 4, 0);
    vision_lm_init_weights(vlm);

    /* No-cache */
    srand(42);
    tensor *img1_s = tensor_randn(ctx.scratch, 4, (int[]){1,3,8,8}, 0);
    int img_n = 3*8*8;
    tensor *img1d = tensor_zeros_data(ctx.data, 4, (int[]){1,3,8,8});
    memcpy(tensor_data_ptr(img1d), tensor_data_ptr(img1_s), (size_t)img_n*sizeof(float));
    int prompt[2] = {0,1};
    tensor *pr1 = make_int_tensor(2, (int[]){1,2}, prompt);
    int n1; int *r1 = vision_lm_generate(ctx.scratch, ctx.data, vlm, img1d, pr1, 10, 0.0f, 0, &n1);
    printf("    nocache: n_out=%d tokens=[", n1);
    for (int i=0; i<n1 && i<10; i++) printf("%d ", r1[i]); printf("]\n");

    /* Cached - reset scratch/data but keep params */
    mem_pool_reset(ctx.scratch);
    mem_pool_reset(ctx.data);

    srand(42);
    tensor *img2_s = tensor_randn(ctx.scratch, 4, (int[]){1,3,8,8}, 0);
    tensor *img2d = tensor_zeros_data(ctx.data, 4, (int[]){1,3,8,8});
    memcpy(tensor_data_ptr(img2d), tensor_data_ptr(img2_s), (size_t)img_n*sizeof(float));
    tensor *pr2 = make_int_tensor(2, (int[]){1,2}, prompt);
    int n2; int *r2 = vision_lm_generate(ctx.scratch, ctx.data, vlm, img2d, pr2, 10, 0.0f, 1, &n2);
    printf("    cached: n_out=%d tokens=[", n2);
    for (int i=0; i<n2 && i<10; i++) printf("%d ", r2[i]); printf("]\n");

    int min = n1<n2 ? n1 : n2;
    assert(min > 0);
    for (int i=0; i<min; i++) assert(r1[i] == r2[i]);
    printf("    OK (both %d tokens, match)\n", min);
}

/* ══════════════════════════════════════════════════════════════════
 *  Test 13: training decreases loss (overfit 1 batch)
 * ══════════════════════════════════════════════════════════════════ */

static void test_training_decreases_loss(void) {
    printf("  test_training_decreases_loss...\n");
    dnn_ctx_init(&ctx, 8*1024*1024, 128*1024*1024, 1*1024*1024);
    srand(42);
    vision_lm *vlm = vision_lm_create(ctx.params, 16, 8, 1, 2, 4, 16,
                                       3, 8, 8, 4, 1);
    vision_lm_init_weights(vlm);
    int n_params;
    tensor **all_p = module_parameters(&vlm->base, &n_params);
    adamw_opt *opt = adamw_create(ctx.params, all_p, n_params, 0.01f,
                                   0.9f, 0.999f, 1e-8f, 0.01f);
    int ids[8] = {0,3,7,5, 1,3,7,5};
    int tgt[8] = {3,7,5,8, 3,7,5,8};
    float first=0, last=0;
    int steps = 20;
    for (int s=0; s<steps; s++) {
        tensor *img = tensor_randn(ctx.scratch, 4, (int[]){2,3,8,8}, 0);
        tensor *inp = make_int_tensor(2, (int[]){2,4}, ids);
        tensor *tar = make_int_tensor(2, (int[]){2,4}, tgt);
        tensor *loss = vision_lm_train_step(ctx.scratch, vlm, img, inp, tar, opt, 0.0f, NULL);
        float lv = tensor_data_ptr(loss)[0];
        if (s==0) first=lv;
        if (s==steps-1) last=lv;
        mem_pool_reset(ctx.scratch);
        mem_pool_reset(ctx.data);
    }
    printf("    first=%.6f last=%.6f\n", first, last);
    assert(last < first && "loss should decrease");
    printf("    OK\n");
}

/* ══════════════════════════════════════════════════════════════════
 *  Test: attention backward dV matches column-sum reference
 *
 *  For causal and prefix-LM modes, compute Q@K^T attention weights
 *  explicitly, then verify that analytical dV = P^T @ ones matches
 *  the reference column sums.  This catches the uninitialized-P bug.
 * ══════════════════════════════════════════════════════════════════ */

static void test_attention_backward_column_sums(void) {
    printf("  test_attention_backward_column_sums...\n");
    float tol = 1e-4f;

    struct { attention_mode mode; int prefix_len; const char *label; } configs[] = {
        {ATTENTION_CAUSAL,    0,  "causal"},
        {ATTENTION_PREFIX_LM, 4,  "prefix-LM"},
    };
    int n_configs = sizeof(configs) / sizeof(configs[0]);

    for (int ci = 0; ci < n_configs; ci++) {
        attention_mode mode = configs[ci].mode;
        int I = configs[ci].prefix_len;
        int B = 1, H = 1, N = 7, d = 4;
        printf("    %s: B=%d H=%d N=%d d=%d I=%d... ", configs[ci].label, B, H, N, d, I);

        dnn_ctx_init(&ctx, 8*1024*1024, 64*1024*1024, 1*1024*1024);
        srand(12345 + ci);

        tensor *Q = tensor_randn(ctx.params, 4, (int[]){B, H, N, d}, 1);
        tensor *K = tensor_randn(ctx.params, 4, (int[]){B, H, N, d}, 1);
        tensor *V = tensor_randn(ctx.params, 4, (int[]){B, H, N, d}, 1);

        /* Explicit column sums of attention weights */
        float *qd = tensor_data_ptr(Q), *kd = tensor_data_ptr(K);
        float scale = 1.0f / sqrtf((float)d);
        double col_sums[7] = {0};
        for (int i = 0; i < N; i++) {
            float scores[7];
            for (int j = 0; j < N; j++) {
                float s = 0;
                for (int k = 0; k < d; k++) s += qd[i*d+k] * kd[j*d+k];
                scores[j] = s * scale;
            }
            int visible = (mode == ATTENTION_CAUSAL) ? i + 1
                         : (i < I) ? I : i + 1;
            if (visible > N) visible = N;
            float mx = -INFINITY;
            for (int j = 0; j < visible; j++) if (scores[j] > mx) mx = scores[j];
            float se = 0;
            for (int j = 0; j < visible; j++) se += expf(scores[j] - mx);
            for (int j = 0; j < visible; j++)
                col_sums[j] += expf(scores[j] - mx) / se;
        }

        /* Analytical dV from the model */
        tensor *O = tensor_attention_ex(ctx.scratch, Q, K, V, NULL, mode, I, NULL);
        tensor *s = O;
        for (int di = 0; di < O->ndim; di++) s = tensor_sum(ctx.scratch, s, 0);
        dnn_backward(ctx.scratch, s);
        float *vg = tensor_grad(V);

        /* dV[b,h,j,d] = sum_i P[i,j] — does not depend on d */
        int n_fails = 0;
        for (int j = 0; j < N && n_fails < 5; j++) {
            float expected = (float)col_sums[j];
            for (int dd = 0; dd < d; dd++) {
                float actual = vg[j * d + dd];
                float err = fabsf(actual - expected);
                if (err > tol && fmaxf(fabsf(actual), fabsf(expected)) > 1e-8f) {
                    float rel = err / fmaxf(fabsf(actual), fabsf(expected));
                    if (rel > tol) {
                        printf("\n      dV[j=%d,d=%d]: anal=%.6f expected=%.6f rel_err=%.6f",
                               j, dd, actual, expected, rel);
                        n_fails++;
                    }
                }
            }
        }
        assert(n_fails == 0 && "dV column-sum mismatch");
        printf("dV matches column sums\n");
        dnn_ctx_destroy(&ctx);
    }
    printf("  Attention column-sum check: all OK\n");
}

/* ══════════════════════════════════════════════════════════════════
 *  Test: tensor_contiguous backward does not corrupt gradient
 *
 *  Checks that the VLM image path:
 *    conv2d -> transpose -> transpose -> tensor_contiguous -> reshape
 *  has correct gradients into the conv.  The bug was that
 *  copy_strided_backward called copy_strided_rec which overwrote
 *  the incoming gradient with parent forward data.
 *
 *  y = contiguous(transpose(x)), loss = sum(y), backward
 *  -> x.grad should be all 1s (transpose is a permutation, contiguous
 *     is a copy, sum backprop is 1 everywhere).
 * ══════════════════════════════════════════════════════════════════ */

static void test_contiguous_backward_correct(void) {
    printf("  test_contiguous_backward_correct... ");
    dnn_ctx_init(&ctx, 8*1024*1024, 64*1024*1024, 1*1024*1024);

    int shape[4] = {2, 3, 4, 5};
    tensor *x = tensor_randn(ctx.params, 4, shape, 1);
    tensor_set_requires_grad(x, 1);

    /* VLM image path: transpose 1<->2 then 2<->3 → noncontiguous → contiguous */
    tensor *t1 = tensor_transpose(ctx.scratch, x, 1, 2);
    tensor *t2 = tensor_transpose(ctx.scratch, t1, 2, 3);
    tensor *c = tensor_contiguous(ctx.scratch, t2);
    /* Sum all elements by reducing dim 0 repeatedly */
    tensor *s = c;
    for (int d = 0; d < c->ndim; d++) s = tensor_sum(ctx.scratch, s, 0);
    tensor *loss = s;

    dnn_backward(ctx.scratch, loss);

    float *xg = tensor_grad(x);
    assert(xg != NULL && "contiguous backward: x.grad is NULL");
    int n = tensor_numel(x);
    for (int i = 0; i < n; i++) {
        if (fabsf(xg[i] - 1.0f) > 1e-4f) {
            printf("FAIL at [%d]: got %.6f, expected 1.0\n", i, xg[i]);
            assert(0 && "contiguous backward: gradient not all ones");
        }
    }
    printf("OK\n");
}

/* ══════════════════════════════════════════════════════════════════
 *  Main
 * ══════════════════════════════════════════════════════════════════ */

int main(void) {
    printf("=== VLM Tests ===\n\n");

    printf("--- Shape tests ---\n");
    test_create_and_shapes();
    test_image_embeds_shape();
    test_build_embeds_shape();
    test_forward_shape();
    test_forward_text_logits_shape();

    printf("\n--- Training tests ---\n");
    test_train_step();
    test_gradients_flow();
    test_training_decreases_loss();

    printf("\n--- Parameter tests ---\n");
    test_param_count();
    test_image_pos_disabled();

    printf("\n--- Attention tests ---\n");
    test_prefix_lm_attention();

    printf("\n--- Generation tests ---\n");
    test_generate_nocache();
    test_generate_cached_matches_nocache();

    printf("\n--- Gradient correctness tests ---\n");
    test_contiguous_backward_correct();
    test_attention_backward_column_sums();

    printf("\n=== All VLM tests passed ===\n");
    return 0;
}
