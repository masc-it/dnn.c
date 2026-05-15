#include "tensor.h"
#include "pool_int.h"
#include "tensor_int.h"
#include "autograd_int.h"
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

static tensor *tensor_copy_strided(struct mem_pool *scratch, const tensor *t) {
    tensor *out = tensor_scratch(scratch, t->ndim, t->shape, t->requires_grad);
    int dst_off = 0;
    copy_strided_rec(t, 0, t->offset, (float*)out->data, &dst_off);
    return out;
}

/* ── Lifecycle ── */

/* contiguous helper: check if strides match packed row-major layout */
static int strides_contiguous(const int *shape, const int *strides, int ndim) {
    int s = 1;
    for (int d = ndim - 1; d >= 0; d--) {
        if (strides[d] != s) return 0;
        s *= shape[d];
    }
    return 1;
}

static tensor *tensor_create_pool(int ndim, const int *shape, struct mem_pool *pool, int requires_grad) {
    assert(pool && "tensor_create_pool: pool is NULL");
    tensor *t = _mem_pool_alloc(pool, sizeof(tensor), NULL);
    t->ndim = ndim;
    memcpy(t->shape, shape, ndim * sizeof(int));
    default_strides(ndim, shape, t->strides);
    int n = tensor_numel_(shape, ndim);
    t->data = _mem_pool_alloc(pool, n * sizeof(float), NULL);
    t->pool = pool;
    t->requires_grad = requires_grad ? 1 : 0;
    t->contiguous = 1;
    t->offset = 0;
    t->parent = NULL;
    t->grad_fn = NULL;
    t->grad = NULL;
    return t;
}

tensor *tensor_scratch(struct mem_pool *pool, int ndim, const int *shape, int requires_grad) {
    assert(pool && "tensor_scratch: pool is NULL");
    tensor *t = _mem_pool_alloc(pool, sizeof(tensor), NULL);
    t->ndim = ndim;
    memcpy(t->shape, shape, ndim * sizeof(int));
    default_strides(ndim, shape, t->strides);
    int n = tensor_numel_(shape, ndim);
    t->data = _mem_pool_alloc_nz(pool, (size_t)n * sizeof(float));
    t->pool = pool;
    t->requires_grad = requires_grad ? 1 : 0;
    t->contiguous = 1;
    t->offset = 0;
    t->parent = NULL;
    t->grad_fn = NULL;
    t->grad = NULL;
    return t;
}

tensor *tensor_zeros(struct mem_pool *pool, int ndim, const int *shape, int requires_grad) {
    return tensor_create_pool(ndim, shape, pool, requires_grad);
}

tensor *tensor_zeros_data(struct mem_pool *pool, int ndim, const int *shape) {
    return tensor_create_pool(ndim, shape, pool, 0);
}

tensor *tensor_randn(struct mem_pool *pool, int ndim, const int *shape, int requires_grad) {
    tensor *t = tensor_create_pool(ndim, shape, pool, requires_grad);
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

tensor *tensor_uniform(struct mem_pool *pool, int ndim, const int *shape, int requires_grad, float bound) {
    tensor *t = tensor_create_pool(ndim, shape, pool, requires_grad);
    int n = tensor_numel_(shape, ndim);
    float *p = (float*)t->data;
    for (int i = 0; i < n; i++) {
        float u = (float)rand() / (float)RAND_MAX;  /* [0, 1) */
        p[i] = (2.0f * u - 1.0f) * bound;           /* [-bound, bound) */
    }
    return t;
}

/* ── slice_backward ── */

static void slice_backward(grad_fn *fn, tensor *grad_output) {
    tensor *parent = fn->inputs[0];
    if (!(parent->grad_fn || parent->requires_grad)) return;

    float *pg = _grad_ensure(parent);
    if (!pg) return;

    int dim   = *(int*)fn->saved_tensors[0];
    int start = *(int*)fn->saved_tensors[1];
    float *gd = (float*)grad_output->data;
    int total = tensor_numel(grad_output);

    int coord[DNN_MAX_DIMS];
    for (int flat = 0; flat < total; flat++) {
        int rem = flat;
        for (int d = grad_output->ndim - 1; d >= 0; d--) {
            coord[d] = rem % grad_output->shape[d];
            rem /= grad_output->shape[d];
        }

        int p_off = parent->offset;
        for (int d = 0; d < parent->ndim; d++) {
            int c = coord[d];
            if (d == dim) c += start;
            p_off += c * parent->strides[d];
        }

        int g_off = grad_output->offset;
        for (int d = 0; d < grad_output->ndim; d++)
            g_off += coord[d] * grad_output->strides[d];

        pg[p_off] += gd[g_off];
    }
}

/* ── reshape_backward ── */

static void reshape_backward(grad_fn *fn, tensor *grad_output) {
    tensor *parent = fn->inputs[0];
    if (!(parent->grad_fn || parent->requires_grad)) return;

    float *pg = _grad_ensure(parent);
    if (!pg) return;

    float *gd = (float*)grad_output->data;
    int total = tensor_numel(grad_output);

    if (grad_output->contiguous && parent->contiguous) {
        int poff = parent->offset;
        int goff = grad_output->offset;
        for (int i = 0; i < total; i++)
            pg[poff + i] += gd[goff + i];
    } else {
        int p_coord[DNN_MAX_DIMS];
        for (int flat = 0; flat < total; flat++) {
            int rem = flat;
            for (int d = grad_output->ndim - 1; d >= 0; d--) {
                p_coord[d] = rem % grad_output->shape[d];
                rem /= grad_output->shape[d];
            }
            int g_off = grad_output->offset;
            int p_off = parent->offset;
            for (int d = 0; d < parent->ndim; d++) {
                g_off += p_coord[d] * grad_output->strides[d];
                p_off += p_coord[d] * parent->strides[d];
            }
            pg[p_off] += gd[g_off];
        }
    }
}

tensor *tensor_slice(struct mem_pool *scratch, tensor *t, int dim, int start, int len) {
    assert(dim < t->ndim && start >= 0 && start + len <= t->shape[dim]);
    tensor *v = _mem_pool_alloc(scratch, sizeof(tensor), t);
    v->shape[dim] = len;
    v->offset = t->offset + start * t->strides[dim];
    v->parent = t;
    v->grad_fn = NULL;
    v->grad = NULL;
    v->pool = NULL;
    v->contiguous = strides_contiguous(v->shape, v->strides, v->ndim);

    /* ── Autograd ── */
    if (dnn_grad_enabled() && tensor_requires_grad(t)) {
        grad_fn *fn = _grad_fn_create(scratch);
        fn->backward = slice_backward;
        fn->n_inputs = 1;
        fn->inputs = _mem_pool_alloc(scratch, 1 * sizeof(tensor*), NULL);
        fn->inputs[0] = t;
        fn->n_saved = 2;
        fn->saved_tensors = _mem_pool_alloc(scratch, 2 * sizeof(tensor*), NULL);
        int *saved_dim   = _mem_pool_alloc(scratch, sizeof(int), NULL);
        int *saved_start = _mem_pool_alloc(scratch, sizeof(int), NULL);
        *saved_dim   = dim;
        *saved_start = start;
        fn->saved_tensors[0] = (tensor*)saved_dim;
        fn->saved_tensors[1] = (tensor*)saved_start;
        v->requires_grad = 1;
        v->grad_fn = fn;
    }

    return v;
}

tensor *tensor_transpose(struct mem_pool *scratch, tensor *t, int d1, int d2) {
    assert(d1 < t->ndim && d2 < t->ndim);
    tensor *v = _mem_pool_alloc(scratch, sizeof(tensor), t);
    v->shape[d1] = t->shape[d2]; v->shape[d2] = t->shape[d1];
    v->strides[d1] = t->strides[d2]; v->strides[d2] = t->strides[d1];
    v->parent = t;
    v->grad_fn = NULL;
    v->grad = NULL;
    v->pool = NULL;
    v->contiguous = strides_contiguous(v->shape, v->strides, v->ndim);
    return v;
}

tensor *tensor_reshape(struct mem_pool *scratch, tensor *t, int ndim, const int *shape) {
    assert(ndim <= DNN_MAX_DIMS && "ndim exceeds DNN_MAX_DIMS");
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
        tensor *v = _mem_pool_alloc(scratch, sizeof(tensor), t);
        v->ndim = ndim;
        memcpy(v->shape, shape, ndim * sizeof(int));
        default_strides(ndim, shape, v->strides);
        v->offset = t->offset;
        v->parent = t;
        v->contiguous = 1;
        v->grad_fn = NULL;
        v->grad = NULL;
        v->pool = NULL;

        if (dnn_grad_enabled() && tensor_requires_grad(t)) {
            grad_fn *fn = _grad_fn_create(scratch);
            fn->backward = reshape_backward;
            fn->n_inputs = 1;
            fn->inputs = _mem_pool_alloc(scratch, 1 * sizeof(tensor*), NULL);
            fn->inputs[0] = t;
            fn->n_saved = 0;
            v->requires_grad = 1;
            v->grad_fn = fn;
        }
        return v;
    }
    return tensor_reshape(scratch, tensor_copy_strided(scratch, t), ndim, shape);
}

tensor *tensor_contiguous(struct mem_pool *scratch, tensor *t) {
    if (tensor_is_contiguous(t)) return t;
    return tensor_copy_strided(scratch, t);
}

tensor *tensor_flatten(struct mem_pool *scratch, tensor *t) {
    return tensor_reshape(scratch, t, 1, (int[]){-1});
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

int tensor_is_leaf(const tensor *t) {
    return t->requires_grad && t->grad_fn == NULL;
}

void tensor_retain_grad(tensor *t) {
    if (!t->grad) {
        tensor *base = t;
        while (base->parent) base = base->parent;
        t->grad = _mem_pool_alloc(base->pool, tensor_numel(t) * sizeof(float), NULL);
    }
}

float *tensor_grad(const tensor *t) {
    return t->grad;
}

tensor *tensor_root(tensor *t) {
    while (t->parent) t = t->parent;
    return t;
}

int tensor_is_contiguous(const tensor *t) {
    return t->contiguous;
}

int tensor_requires_grad(const tensor *t) {
    return t->requires_grad;
}

void tensor_set_requires_grad(tensor *t, int req) {
    t->requires_grad = req ? 1 : 0;
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
