#include "dnn.h"
#include "attention.h"
#include <stdio.h>
#include <assert.h>
#include <math.h>
#include <string.h>

#define EPS 1e-4f

/* helper: check tensor data elements */
static void check_data_ary(tensor *t, const float *exp, int n, const char *label) {
    float *d = tensor_data_ptr(t);
    for (int i = 0; i < n; i++) {
        if (fabsf(d[i] - exp[i]) > EPS) {
            printf("    FAIL: %s[%d]: got %.6f, expected %.6f\n", label, i, d[i], exp[i]);
            assert(0);
        }
    }
}

/* helper: check tensor data with custom tolerance */
static void check_data_ary_tol(tensor *t, const float *exp, int n, float tol, const char *label) {
    float *d = tensor_data_ptr(t);
    for (int i = 0; i < n; i++) {
        if (fabsf(d[i] - exp[i]) > tol) {
            printf("    FAIL: %s[%d]: got %.6f, expected %.6f (tol=%.6f)\n", label, i, d[i], exp[i], tol);
            assert(0);
        }
    }
}

/* ── Helper: uniform tensor from a float array ── */
static tensor *make_tensor(int ndim, const int *shape, const float *data, int rg) {
    tensor *t = tensor_zeros(ndim, shape, rg);
    float *td = tensor_data_ptr(t);
    int n = tensor_numel(t);
    memcpy(td, data, n * sizeof(float));
    return t;
}

/* ── Test 1: 2D attention forward (single sequence, no batch) ──
 *
 *   N=2, d_k=2.  Known values:
 *     Q = [[1, 0],   K = [[1, 2],   V = [[1, 1],
 *          [0, 1]]        [3, 4]]        [2, 2]]
 *
 *   scores = Q @ K^T / sqrt(2)
 *     K^T = [[1, 3],
 *            [2, 4]]
 *     Q @ K^T = [[1*1+0*2, 1*3+0*4], [0*1+1*2, 0*3+1*4]] = [[1, 3], [2, 4]]
 *     scores = [[1/√2, 3/√2], [2/√2, 4/√2]] ≈ [[0.7071, 2.1213], [1.4142, 2.8284]]
 *
 *   softmax row 0: max=2.1213, exp(x-max)=[exp(-1.4142)=0.2429, exp(0)=1], sum=1.2429
 *     softmax = [0.2429/1.2429=0.1954, 1/1.2429=0.8046]
 *   softmax row 1: max=2.8284, exp(x-max)=[exp(-1.4142)=0.2429, exp(0)=1], sum=1.2429
 *     softmax = [0.2429/1.2429=0.1954, 0.8046]
 *
 *   output = attn @ V = softmax @ [[1,1],[2,2]]
 *     row 0: 0.1954*[1,1] + 0.8046*[2,2] = [0.1954+1.6092, 0.1954+1.6092] = [1.8046, 1.8046]
 *     row 1: same = [1.8046, 1.8046]
 */
static void test_attention_2d_forward(void) {
    printf("  test_attention_2d_forward... ");
    float q_data[] = {1.0f, 0.0f, 0.0f, 1.0f};
    float k_data[] = {1.0f, 2.0f, 3.0f, 4.0f};
    float v_data[] = {1.0f, 1.0f, 2.0f, 2.0f};
    tensor *Q = make_tensor(2, (int[]){2, 2}, q_data, 0);
    tensor *K = make_tensor(2, (int[]){2, 2}, k_data, 0);
    tensor *V = make_tensor(2, (int[]){2, 2}, v_data, 0);

    tensor *out = tensor_attention(Q, K, V, NULL);
    assert(tensor_ndim(out) == 2);
    assert(tensor_shape(out, 0) == 2);
    assert(tensor_shape(out, 1) == 2);

    /* expected computed above */
    float exp[] = {1.8046f, 1.8046f, 1.8046f, 1.8046f};
    check_data_ary_tol(out, exp, 4, 1e-3f, "out");
    printf("OK\n");
}

/* ── Test 2: 2D attention with causal mask ──
 *
 *   N=2, d_k=2. Same Q, K, V, but with causal mask (triu(N,1)).
 *   mask = [[0, -inf], [0, 0]]
 *
 *   scores = Q @ K^T / sqrt(2) = [[0.7071, 2.1213], [1.4142, 2.8284]]
 *   + mask = [[0.7071, -inf], [1.4142, 2.8284]]
 *
 *   softmax row 0: only position 0 has value → [1, 0]
 *   softmax row 1: same as before → [0.1954, 0.8046]
 *
 *   output = attn @ V:
 *     row 0: 1*[1,1] + 0*[2,2] = [1, 1]
 *     row 1: 0.1954*[1,1] + 0.8046*[2,2] = [1.8046, 1.8046]
 */
static void test_attention_2d_causal(void) {
    printf("  test_attention_2d_causal... ");
    float q_data[] = {1.0f, 0.0f, 0.0f, 1.0f};
    float k_data[] = {1.0f, 2.0f, 3.0f, 4.0f};
    float v_data[] = {1.0f, 1.0f, 2.0f, 2.0f};
    tensor *Q = make_tensor(2, (int[]){2, 2}, q_data, 0);
    tensor *K = make_tensor(2, (int[]){2, 2}, k_data, 0);
    tensor *V = make_tensor(2, (int[]){2, 2}, v_data, 0);
    tensor *mask = tensor_triu(2, 1);

    tensor *out = tensor_attention(Q, K, V, mask);
    assert(tensor_ndim(out) == 2);
    assert(tensor_shape(out, 0) == 2);
    assert(tensor_shape(out, 1) == 2);

    float exp[] = {1.0f, 1.0f, 1.8046f, 1.8046f};
    check_data_ary_tol(out, exp, 4, 1e-3f, "out");
    printf("OK\n");
}

/* ── Test 3: 3D batched attention (batch=2, N=2, d=2) ──
 *
 * Same 2D test repeated for 2 batch items, with different values.
 */
static void test_attention_3d_forward(void) {
    printf("  test_attention_3d_forward... ");
    /* batch 0: same as 2D test
     * batch 1: Q=[[2,0],[0,2]], K same, V=2× [[1,1],[2,2]]
     */
    float q_data[] = {1.0f, 0.0f, 0.0f, 1.0f,  /* batch 0 */
                      2.0f, 0.0f, 0.0f, 2.0f}; /* batch 1 */
    float k_data[] = {1.0f, 2.0f, 3.0f, 4.0f,
                      1.0f, 2.0f, 3.0f, 4.0f};
    float v_data[] = {1.0f, 1.0f, 2.0f, 2.0f,
                      1.0f, 1.0f, 2.0f, 2.0f};
    tensor *Q = make_tensor(3, (int[]){2, 2, 2}, q_data, 0);
    tensor *K = make_tensor(3, (int[]){2, 2, 2}, k_data, 0);
    tensor *V = make_tensor(3, (int[]){2, 2, 2}, v_data, 0);

    tensor *out = tensor_attention(Q, K, V, NULL);
    assert(tensor_ndim(out) == 3);
    assert(tensor_shape(out, 0) == 2);
    assert(tensor_shape(out, 1) == 2);
    assert(tensor_shape(out, 2) == 2);

    /* batch 0: same as 2D test
     * batch 1: Q*2, so scores are 2x, softmax changes slightly
     * scores = Q @ K^T / sqrt(2)
     *   batch 1: Q=[2,0; 0,2], same K, scores = [[2*1+0*2=2, 2*3+0*4=6] / 1.414 = [1.4142, 4.2426],
     *                                           [0*1+2*2=4, 0*3+2*4=8] / 1.414 = [2.8284, 5.6568]]
     *   softmax row 0: max=4.2426, exp=[exp(-2.8284)=0.0592, exp(0)=1], sum=1.0592 → [0.0559, 0.9441]
     *   row 1: max=5.6568, exp=[exp(-2.8284)=0.0592, exp(0)=1], sum=1.0592 → [0.0559, 0.9441]
     *   output = attn @ V:
     *     row 0: 0.0559*[1,1] + 0.9441*[2,2] = [1.9441, 1.9441]
     *     row 1: same = [1.9441, 1.9441]
     */
    float exp[] = {1.8046f, 1.8046f, 1.8046f, 1.8046f,
                   1.9441f, 1.9441f, 1.9441f, 1.9441f};
    check_data_ary_tol(out, exp, 8, 1e-3f, "out");
    printf("OK\n");
}

/* ── Test 4: 2D attention with grad check ──
 *
 * Simple 2D case, verify backward produces non-NULL grads.
 * N=2, d_k=2.
 */
static void test_attention_2d_backward(void) {
    printf("  test_attention_2d_backward... ");
    float q_data[] = {1.0f, 0.0f, 0.0f, 1.0f};
    float k_data[] = {1.0f, 2.0f, 3.0f, 4.0f};
    float v_data[] = {1.0f, 1.0f, 2.0f, 2.0f};
    tensor *Q = make_tensor(2, (int[]){2, 2}, q_data, 1);
    tensor *K = make_tensor(2, (int[]){2, 2}, k_data, 1);
    tensor *V = make_tensor(2, (int[]){2, 2}, v_data, 1);

    tensor *out = tensor_attention(Q, K, V, NULL);
    tensor *loss = tensor_sum(out, -1);  /* sum over last dim */
    dnn_backward(loss);

    /* all three inputs should have grads */
    assert(tensor_grad(Q) && "dQ");
    assert(tensor_grad(K) && "dK");
    assert(tensor_grad(V) && "dV");

    /* grads should be non-zero */
    int nq = tensor_numel(Q);
    int nk = tensor_numel(K);
    int nv = tensor_numel(V);
    float *gq = tensor_grad(Q);
    float *gk = tensor_grad(K);
    float *gv = tensor_grad(V);
    for (int i = 0; i < nq; i++)
        assert(fabsf(gq[i]) > 0 && "dQ must be non-zero");
    for (int i = 0; i < nk; i++)
        assert(fabsf(gk[i]) > 0 && "dK must be non-zero");
    for (int i = 0; i < nv; i++)
        assert(fabsf(gv[i]) > 0 && "dV must be non-zero");

    printf("OK\n");
}

/* ── Test 5: Attention with causal mask + backward ── */
static void test_attention_causal_backward(void) {
    printf("  test_attention_causal_backward... ");
    float q_data[] = {1.0f, 0.0f, 0.0f, 1.0f};
    float k_data[] = {1.0f, 2.0f, 3.0f, 4.0f};
    float v_data[] = {1.0f, 1.0f, 2.0f, 2.0f};
    tensor *Q = make_tensor(2, (int[]){2, 2}, q_data, 1);
    tensor *K = make_tensor(2, (int[]){2, 2}, k_data, 1);
    tensor *V = make_tensor(2, (int[]){2, 2}, v_data, 1);
    tensor *mask = tensor_triu(2, 1);

    tensor *out = tensor_attention(Q, K, V, mask);
    tensor *loss = tensor_sum(out, -1);
    dnn_backward(loss);

    assert(tensor_grad(Q) && "dQ with mask");
    assert(tensor_grad(K) && "dK with mask");
    assert(tensor_grad(V) && "dV with mask");
    printf("OK\n");
}

/* ── Test 6: 4D multi-head attention ──
 *
 * B=1, H=2, N=2, d_k=2.  Two heads with different Q values.
 */
static void test_attention_4d_forward(void) {
    printf("  test_attention_4d_forward... ");
    /* head 0: Q identity (same as 2D test)
     * head 1: Q = 2*identity
     * K, V shared across heads (same data for both heads) */
    float q_data[] = {1.0f, 0.0f, 0.0f, 1.0f,  /* head 0 */
                      2.0f, 0.0f, 0.0f, 2.0f}; /* head 1 */
    float k_data[] = {1.0f, 2.0f, 3.0f, 4.0f,
                      1.0f, 2.0f, 3.0f, 4.0f};
    float v_data[] = {1.0f, 1.0f, 2.0f, 2.0f,
                      1.0f, 1.0f, 2.0f, 2.0f};
    tensor *Q = make_tensor(4, (int[]){1, 2, 2, 2}, q_data, 0);
    tensor *K = make_tensor(4, (int[]){1, 2, 2, 2}, k_data, 0);
    tensor *V = make_tensor(4, (int[]){1, 2, 2, 2}, v_data, 0);

    tensor *out = tensor_attention(Q, K, V, NULL);
    assert(tensor_ndim(out) == 4);
    assert(tensor_shape(out, 0) == 1);
    assert(tensor_shape(out, 1) == 2);
    assert(tensor_shape(out, 2) == 2);
    assert(tensor_shape(out, 3) == 2);

    /* head 0 output = same as 2D test [1.8046, 1.8046; 1.8046, 1.8046]
     * head 1 output = same as batch 1 in 3D test [1.9441, 1.9441; 1.9441, 1.9441] */
    float exp[] = {1.8046f, 1.8046f, 1.8046f, 1.8046f,
                   1.9441f, 1.9441f, 1.9441f, 1.9441f};
    check_data_ary_tol(out, exp, 8, 1e-3f, "out");
    printf("OK\n");
}

/* ── Test 7: Attention backward with 4D inputs ── */
static void test_attention_4d_backward(void) {
    printf("  test_attention_4d_backward... ");
    float q_data[] = {1.0f, 0.0f, 0.0f, 1.0f,
                      2.0f, 0.0f, 0.0f, 2.0f};
    float k_data[] = {1.0f, 2.0f, 3.0f, 4.0f,
                      1.0f, 2.0f, 3.0f, 4.0f};
    float v_data[] = {1.0f, 1.0f, 2.0f, 2.0f,
                      1.0f, 1.0f, 2.0f, 2.0f};
    tensor *Q = make_tensor(4, (int[]){1, 2, 2, 2}, q_data, 1);
    tensor *K = make_tensor(4, (int[]){1, 2, 2, 2}, k_data, 1);
    tensor *V = make_tensor(4, (int[]){1, 2, 2, 2}, v_data, 1);

    tensor *out = tensor_attention(Q, K, V, NULL);
    tensor *loss = tensor_sum(out, -1);
    dnn_backward(loss);

    assert(tensor_grad(Q) && "dQ 4D");
    assert(tensor_grad(K) && "dK 4D");
    assert(tensor_grad(V) && "dV 4D");
    printf("OK\n");
}

/* ── Test 8: Dim=1, N=1 (edge case: single token) ──
 *
 * Single position: attention is identity (softmax of 1 element = 1)
 * output = V (since attn = [1])
 */
static void test_attention_single_token(void) {
    printf("  test_attention_single_token... ");
    float q_data[] = {5.0f};
    float k_data[] = {3.0f};
    float v_data[] = {7.0f};
    tensor *Q = make_tensor(2, (int[]){1, 1}, q_data, 0);
    tensor *K = make_tensor(2, (int[]){1, 1}, k_data, 0);
    tensor *V = make_tensor(2, (int[]){1, 1}, v_data, 0);

    tensor *out = tensor_attention(Q, K, V, NULL);
    /* score = 5*3/1 = 15, softmax([15]) = [1], output = 1*7 = 7 */
    float exp[] = {7.0f};
    check_data_ary(out, exp, 1, "out");
    printf("OK\n");
}

/* ── Test 9: Verify attention allocates no heap during training ──
 *
 * (Structural test — ensure grad_fn is set on output when inputs require grad)
 */
static void test_attention_tape_wired(void) {
    printf("  test_attention_tape_wired... ");
    float q_data[] = {1.0f, 0.0f, 0.0f, 1.0f};
    float k_data[] = {1.0f, 2.0f, 3.0f, 4.0f};
    float v_data[] = {1.0f, 1.0f, 2.0f, 2.0f};
    tensor *Q = make_tensor(2, (int[]){2, 2}, q_data, 1);
    tensor *K = make_tensor(2, (int[]){2, 2}, k_data, 0);
    tensor *V = make_tensor(2, (int[]){2, 2}, v_data, 0);

    tensor *out = tensor_attention(Q, K, V, NULL);
    assert(out->grad_fn != NULL && "attention must create grad_fn when any input requires_grad");
    printf("OK\n");
}

/* ── Test 10: No-grad mode ── */
static void test_attention_no_grad(void) {
    printf("  test_attention_no_grad... ");
    float q_data[] = {1.0f, 0.0f, 0.0f, 1.0f};
    float k_data[] = {1.0f, 2.0f, 3.0f, 4.0f};
    float v_data[] = {1.0f, 1.0f, 2.0f, 2.0f};
    tensor *Q = make_tensor(2, (int[]){2, 2}, q_data, 1);
    tensor *K = make_tensor(2, (int[]){2, 2}, k_data, 1);
    tensor *V = make_tensor(2, (int[]){2, 2}, v_data, 1);

    dnn_grad_ctx ctx = dnn_no_grad_enter();
    tensor *out = tensor_attention(Q, K, V, NULL);
    dnn_no_grad_exit(ctx);

    assert(out->grad_fn == NULL && "no grad_fn in no_grad mode");
    printf("OK\n");
}

/* ── Test 11: 3D batched backward ── */
static void test_attention_3d_backward(void) {
    printf("  test_attention_3d_backward... ");
    float q_data[] = {1.0f, 0.0f, 0.0f, 1.0f,
                      2.0f, 0.0f, 0.0f, 2.0f};
    float k_data[] = {1.0f, 2.0f, 3.0f, 4.0f,
                      1.0f, 2.0f, 3.0f, 4.0f};
    float v_data[] = {1.0f, 1.0f, 2.0f, 2.0f,
                      1.0f, 1.0f, 2.0f, 2.0f};
    tensor *Q = make_tensor(3, (int[]){2, 2, 2}, q_data, 1);
    tensor *K = make_tensor(3, (int[]){2, 2, 2}, k_data, 1);
    tensor *V = make_tensor(3, (int[]){2, 2, 2}, v_data, 1);
    tensor *mask = tensor_triu(2, 1);

    tensor *out = tensor_attention(Q, K, V, mask);
    tensor *loss = tensor_sum(out, -1);
    dnn_backward(loss);

    assert(tensor_grad(Q) && "dQ 3D batch");
    assert(tensor_grad(K) && "dK 3D batch");
    assert(tensor_grad(V) && "dV 3D batch");
    printf("OK\n");
}

int main(void) {
    printf("test_attention:\n");

    mem_pool params  = mem_pool_create(10 * 1024 * 1024);
    mem_pool scratch = mem_pool_create(50 * 1024 * 1024);
    mem_pool_set_defaults(&params, &scratch, NULL);

    test_attention_2d_forward();
    mem_pool_reset(&params);
    mem_pool_reset(&scratch);

    test_attention_2d_causal();
    mem_pool_reset(&params);
    mem_pool_reset(&scratch);

    test_attention_3d_forward();
    mem_pool_reset(&params);
    mem_pool_reset(&scratch);

    test_attention_2d_backward();
    mem_pool_reset(&params);
    mem_pool_reset(&scratch);

    test_attention_causal_backward();
    mem_pool_reset(&params);
    mem_pool_reset(&scratch);

    test_attention_4d_forward();
    mem_pool_reset(&params);
    mem_pool_reset(&scratch);

    test_attention_4d_backward();
    mem_pool_reset(&params);
    mem_pool_reset(&scratch);

    test_attention_single_token();
    mem_pool_reset(&params);
    mem_pool_reset(&scratch);

    test_attention_tape_wired();
    mem_pool_reset(&params);
    mem_pool_reset(&scratch);

    test_attention_no_grad();
    mem_pool_reset(&params);
    mem_pool_reset(&scratch);

    test_attention_3d_backward();
    mem_pool_reset(&params);
    mem_pool_reset(&scratch);

    mem_pool_destroy(&params);
    mem_pool_destroy(&scratch);

    printf("  ALL PASS\n");
    return 0;
}
