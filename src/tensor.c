#include "tensor.h"
#include "_internal.h"
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <stdio.h>
#include <assert.h>

/* ── internal helpers ── */

static void default_strides(int ndim, const int *shape, int *strides) {
    int s = 1;
    for (int i = ndim - 1; i >= 0; i--) {
        strides[i] = s;
        s *= shape[i];
    }
}

static int tensor_numel_(const int *shape, int ndim) {
    int n = 1;
    for (int i = 0; i < ndim; i++) n *= shape[i];
    return n;
}

/* stride-aware copy: walk logical order, write contiguous */
static void copy_strided_rec(const tensor *src, int dim, int src_off, float *dst, int *dst_off) {
    if (dim == src->ndim - 1) {
        int n = src->shape[dim];
        float *s = (float*)src->data;
        int stride = src->strides[dim];
        for (int i = 0; i < n; i++)
            dst[(*dst_off)++] = s[src_off + i * stride];
    } else {
        for (int i = 0; i < src->shape[dim]; i++)
            copy_strided_rec(src, dim + 1, src_off + i * src->strides[dim], dst, dst_off);
    }
}

static tensor *tensor_copy_strided(const tensor *t) {
    tensor *out = tensor_scratch_create_(t->ndim, t->shape, 0);
    int dst_off = 0;
    copy_strided_rec(t, 0, t->offset, (float*)out->data, &dst_off);
    return out;
}

/* ── Lifecycle ── */

static tensor *tensor_create_pool(int ndim, const int *shape, mem_pool *pool, int requires_grad) {
    assert(pool && "tensor_create_pool: default pool not set");
    tensor *t = mem_pool_alloc(pool, sizeof(tensor), NULL);
    t->ndim = ndim;
    memcpy(t->shape, shape, ndim * sizeof(int));
    default_strides(ndim, shape, t->strides);
    int n = tensor_numel_(shape, ndim);
    t->data = mem_pool_alloc(pool, n * sizeof(float), NULL);
    t->pool = pool;
    if (requires_grad) tensor_set_requires_grad(t, 1);
    return t;
}

tensor *tensor_scratch_create_(int ndim, const int *shape, int requires_grad) {
    return tensor_create_pool(ndim, shape, mem_pool_scratch(), requires_grad);
}

tensor *tensor_zeros(int ndim, const int *shape, int requires_grad) {
    return tensor_create_pool(ndim, shape, mem_pool_params(), requires_grad);
}

tensor *tensor_randn(int ndim, const int *shape, int requires_grad) {
    tensor *t = tensor_create_pool(ndim, shape, mem_pool_params(), requires_grad);
    int n = tensor_numel_(shape, ndim);
    float *p = (float*)t->data;
    for (int i = 0; i < n; i += 2) {
        float u1 = (float)rand() / (float)RAND_MAX;
        float u2 = (float)rand() / (float)RAND_MAX;
        float r = sqrtf(-2.0f * logf(u1 + 1e-10f));
        float theta = 6.283185307179586f * u2;  /* 2*pi */
        p[i] = r * cosf(theta);
        if (i + 1 < n)
            p[i + 1] = r * sinf(theta);
    }
    return t;
}

/* ── Views ── */

tensor *tensor_slice(tensor *t, int dim, int start, int len) {
    assert(dim < t->ndim && start >= 0 && start + len <= t->shape[dim]);
    tensor *v = mem_scratch_alloc(sizeof(tensor), t);
    v->shape[dim] = len;
    v->offset = t->offset + start * t->strides[dim];
    v->parent = t;
    v->grad_fn = NULL;
    v->grad = NULL;
    v->pool = NULL;
    return v;
}

tensor *tensor_transpose(tensor *t, int d1, int d2) {
    assert(d1 < t->ndim && d2 < t->ndim);
    tensor *v = mem_scratch_alloc(sizeof(tensor), t);
    v->shape[d1] = t->shape[d2]; v->shape[d2] = t->shape[d1];
    v->strides[d1] = t->strides[d2]; v->strides[d2] = t->strides[d1];
    v->parent = t;
    v->grad_fn = NULL;
    v->grad = NULL;
    v->pool = NULL;
    return v;
}

tensor *tensor_reshape(tensor *t, int ndim, const int *shape) {
    assert(ndim <= DNN_MAX_DIMS && "ndim exceeds DNN_MAX_DIMS");
    /* resolve -1: infer the missing dimension */
    int resolved[DNN_MAX_DIMS];
    int inferred = -1;
    int product  = 1;
    for (int i = 0; i < ndim; i++) {
        if (shape[i] == -1) {
            assert(inferred == -1 && "only one -1 allowed");
            inferred = i;
        } else {
            assert(shape[i] > 0 && "dims must be positive or -1");
            product *= shape[i];
        }
    }
    int total = tensor_numel(t);
    if (inferred >= 0) {
        assert(total % product == 0 && "-1 dim must divide evenly");
        int inferred_val = total / product;
        memcpy(resolved, shape, ndim * sizeof(int));
        resolved[inferred] = inferred_val;
        shape = resolved;
    } else {
        assert(product == total && "shape must match numel");
    }

    if (tensor_is_contiguous(t)) {
        /* view: keep parent, override shape/strides */
        tensor *v = mem_scratch_alloc(sizeof(tensor), t);
        v->ndim = ndim;
        memcpy(v->shape, shape, ndim * sizeof(int));
        default_strides(ndim, shape, v->strides);
        v->offset = t->offset;
        v->parent = t;
        v->grad_fn = NULL;
        v->grad = NULL;
        v->pool = NULL;
        return v;
    }
    /* non-contiguous: copy to new contiguous tensor with new shape */
    return tensor_reshape(tensor_copy_strided(t), ndim, shape);
}

tensor *tensor_contiguous(tensor *t) {
    if (tensor_is_contiguous(t)) return t;
    return tensor_copy_strided(t);
}

tensor *tensor_flatten(tensor *t) {
    return tensor_reshape(t, 1, (int[]){-1});
}

/* ── Accessors ── */

float *tensor_data_ptr(tensor *t) {
    assert(tensor_is_contiguous(t));
    return (float*)t->data + t->offset;
}

int tensor_numel(const tensor *t) {
    return tensor_numel_(t->shape, t->ndim);
}

int tensor_ndim(const tensor *t) {
    return t->ndim;
}

int tensor_shape(const tensor *t, int dim) {
    return t->shape[dim];
}

/* ── Properties ── */

int tensor_is_contiguous(const tensor *t) {
    int s = 1;
    for (int i = t->ndim - 1; i >= 0; i--) {
        if (t->strides[i] != s) return 0;
        s *= t->shape[i];
    }
    return 1;
}

int tensor_requires_grad(const tensor *t) {
    return t->grad != NULL;
}

void tensor_set_requires_grad(tensor *t, int req) {
    if (req && !t->grad) {
        int n = tensor_numel_(t->shape, t->ndim);
        t->grad = mem_params_alloc(n * sizeof(float), NULL);
    } else if (!req && t->grad) {
        /* Pool owns grad — just drop the pointer */
        t->grad = NULL;
    }
}

void tensor_print(const tensor *t) {
    printf("tensor(shape=[");
    for (int i = 0; i < t->ndim; i++)
        printf("%s%d", i ? "," : "", t->shape[i]);
    printf("], strides=[");
    for (int i = 0; i < t->ndim; i++)
        printf("%s%d", i ? "," : "", t->strides[i]);
    int n = tensor_numel_(t->shape, t->ndim);
    int show = n < 16 ? n : 16;
    printf("], data=[");
    for (int i = 0; i < show; i++)
        printf("%s%.4f", i ? "," : "", ((float*)t->data)[t->offset + i]);
    if (n > 16) printf(",...");
    printf("])\n");
}
