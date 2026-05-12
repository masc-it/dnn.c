#include "ops.h"
#include "_internal.h"
#include <stdlib.h>
#include <assert.h>

/* ── Arithmetic (stubs) ── */

tensor *tensor_add(const tensor *a, const tensor *b) {
    (void)a; (void)b;
    /* TODO: broadcast + element-wise add + autograd tape */
    return NULL;
}

tensor *tensor_sub(const tensor *a, const tensor *b) {
    (void)a; (void)b;
    return NULL;
}

tensor *tensor_mul(const tensor *a, const tensor *b) {
    (void)a; (void)b;
    return NULL;
}

tensor *tensor_div(const tensor *a, const tensor *b) {
    (void)a; (void)b;
    return NULL;
}

/* ── Matrix ops ── */

tensor *tensor_matmul(const tensor *a, const tensor *b) {
    (void)a; (void)b;
    return NULL;
}

/* ── Activations ── */

tensor *tensor_relu(const tensor *t) {
    (void)t;
    return NULL;
}

tensor *tensor_sigmoid(const tensor *t) {
    (void)t;
    return NULL;
}

tensor *tensor_tanh(const tensor *t) {
    (void)t;
    return NULL;
}

/* ── Reduction ── */

tensor *tensor_sum(const tensor *t, int dim) {
    (void)t; (void)dim;
    return NULL;
}

tensor *tensor_mean(const tensor *t, int dim) {
    (void)t; (void)dim;
    return NULL;
}

/* ── Utility ── */

tensor *tensor_pow(const tensor *t, float exp) {
    (void)t; (void)exp;
    return NULL;
}

tensor *tensor_neg(const tensor *t) {
    (void)t;
    return NULL;
}
