#include "ops.h"
#include "pool.h"
#include <stdio.h>
#include <assert.h>
#include <math.h>

/* ── Helpers ── */

static int is_neginf(float v) { return v == -INFINITY; }

static void print_mask(tensor *t) {
    int N = tensor_shape(t, 0);
    float *d = (float*)t->data + t->offset;
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            float v = d[i * N + j];
            if (isinf(v) && v < 0) printf(" -inf");
            else if (v == 0.0f)    printf("    0");
            else                   printf(" %5.1f", v);
        }
        printf("\n");
    }
}

/* ── tensor_triu formula ──
 *
 * Element (i,j) = -INFINITY if j >= i + diagonal, 0.0f otherwise.
 *
 *   diagonal=1 (standard causal): j >= i+1 → -inf
 *     j <= i → 0, j > i → -inf → self+past kept, future masked
 *
 *   diagonal=0 (no self-attention): j >= i → -inf
 *     j < i → 0, j >= i → -inf → only past kept
 */

/* ── Test 1: N=1, diagonal=0 — no self-attention ── */
static void test_triu_n1_d0(void) {
    /* j >= i + 0 → j >= i
     * (0,0): 0 >= 0 → T → -inf */
    tensor *m = tensor_triu(1, 0);
    float *d = (float*)m->data + m->offset;
    assert(is_neginf(d[0]));
    mem_pool_reset(m->pool);
}

/* ── Test 2: N=1, diagonal=1 — standard causal ── */
static void test_triu_n1_d1(void) {
    /* j >= i + 1
     * (0,0): 0 >= 1 → F → 0 */
    tensor *m = tensor_triu(1, 1);
    float *d = (float*)m->data + m->offset;
    assert(d[0] == 0.0f);
    mem_pool_reset(m->pool);
}

/* ── Test 3: N=3, diagonal=0 — no self-attention ── */
static void test_triu_n3_d0(void) {
    tensor *m = tensor_triu(3, 0);
    float *d = (float*)m->data + m->offset;
    /* row 0: j >= 0 → all -inf */
    assert(is_neginf(d[0]) && is_neginf(d[1]) && is_neginf(d[2]));
    /* row 1: j >= 1 → col 0 is 0, col 1-2 -inf */
    assert(d[3] == 0.0f && is_neginf(d[4]) && is_neginf(d[5]));
    /* row 2: j >= 2 → col 0-1 are 0, col 2 -inf */
    assert(d[6] == 0.0f && d[7] == 0.0f && is_neginf(d[8]));
    mem_pool_reset(m->pool);
}

/* ── Test 4: N=3, diagonal=1 — standard causal mask ── */
static void test_triu_n3_d1(void) {
    /* j >= i+1 → -inf, else 0
     *   (0,0): 0 → 0  (0,1): 1 → -inf  (0,2): 2 → -inf
     *   (1,0): 0 → 0  (1,1): 1 ≥ 2? N → 0  (1,2): 2 → -inf
     *   (2,0): 0 → 0  (2,1): 1 → 0           (2,2): 2 ≥ 3? N → 0
     */
    tensor *m = tensor_triu(3, 1);
    float *d = (float*)m->data + m->offset;
    assert(d[0] == 0.0f && is_neginf(d[1]) && is_neginf(d[2]));
    assert(d[3] == 0.0f && d[4] == 0.0f && is_neginf(d[5]));
    assert(d[6] == 0.0f && d[7] == 0.0f && d[8] == 0.0f);
    mem_pool_reset(m->pool);
}

/* ── Test 5: N=3, diagonal=2 — attend further into past ── */
static void test_triu_n3_d2(void) {
    /* j >= i+2 → -inf
     *   (0,0): 0 → 0  (0,1): 1 → 0  (0,2): 2 → -inf
     *   (1,0): 0 → 0  (1,1): 1 → 0  (1,2): 2 → 0
     *   (2,0): 0 → 0  (2,1): 1 → 0  (2,2): 2 → 0
     */
    tensor *m = tensor_triu(3, 2);
    float *d = (float*)m->data + m->offset;
    assert(d[0] == 0.0f && d[1] == 0.0f && is_neginf(d[2]));
    assert(d[3] == 0.0f && d[4] == 0.0f && d[5] == 0.0f);
    assert(d[6] == 0.0f && d[7] == 0.0f && d[8] == 0.0f);
    mem_pool_reset(m->pool);
}

/* ── Test 6: N=4, diagonal=1 full verification ── */
static void test_triu_n4_d1(void) {
    tensor *m = tensor_triu(4, 1);
    float *d = (float*)m->data + m->offset;
    int expected[16] = {
    /*  j=0  j=1  j=2  j=3 */
        0,   1,   1,   1,   /* i=0 */
        0,   0,   1,   1,   /* i=1 */
        0,   0,   0,   1,   /* i=2 */
        0,   0,   0,   0    /* i=3 */
    };
    for (int i = 0; i < 16; i++) {
        if (expected[i])
            assert(is_neginf(d[i]));
        else
            assert(d[i] == 0.0f);
    }
    mem_pool_reset(m->pool);
}

/* ── Test 7: N=5, diagonal=1 smoke ── */
static void test_triu_n5_d1(void) {
    tensor *m = tensor_triu(5, 1);
    float *d = (float*)m->data + m->offset;
    for (int i = 0; i < 5; i++) {
        for (int j = 0; j < 5; j++) {
            int idx = i * 5 + j;
            if (j > i)
                assert(is_neginf(d[idx]));
            else
                assert(d[idx] == 0.0f);
        }
    }
    mem_pool_reset(m->pool);
}

/* ── Test 8: structural properties ── */
static void test_triu_properties(void) {
    tensor *m = tensor_triu(4, 1);
    assert(tensor_ndim(m) == 2);
    assert(tensor_shape(m, 0) == 4);
    assert(tensor_shape(m, 1) == 4);
    assert(tensor_numel(m) == 16);
    assert(!tensor_requires_grad(m));
    assert(m->grad_fn == NULL);
    mem_pool_reset(m->pool);
}

int main(void) {
    printf("test_triu: causal mask via tensor_triu\n");

    mem_pool params  = mem_pool_create(64 * 1024);
    mem_pool scratch = mem_pool_create(64 * 1024);
    mem_pool_set_defaults(&params, &scratch, NULL);

    printf("  triu N=1 diag=0... ");   test_triu_n1_d0();    printf("OK\n");
    printf("  triu N=1 diag=1... ");   test_triu_n1_d1();    printf("OK\n");
    printf("  triu N=3 diag=0... ");   test_triu_n3_d0();    printf("OK\n");
    printf("  triu N=3 diag=1... ");   test_triu_n3_d1();    printf("OK\n");
    printf("  triu N=3 diag=2... ");   test_triu_n3_d2();    printf("OK\n");
    printf("  triu N=4 diag=1... ");   test_triu_n4_d1();    printf("OK\n");
    printf("  triu N=5 diag=1... ");   test_triu_n5_d1();    printf("OK\n");
    printf("  triu properties... ");    test_triu_properties(); printf("OK\n");

    mem_pool_destroy(&params);
    mem_pool_destroy(&scratch);

    printf("  PASS\n");
    return 0;
}
