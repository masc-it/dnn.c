#include "autograd.h"
#include "pool.h"
#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <assert.h>


/* ── Grad mode ── */

static _Thread_local int _grad_enabled = 1;

int dnn_grad_enabled(void) {
    return _grad_enabled;
}

dnn_grad_ctx dnn_no_grad_enter(void) {
    dnn_grad_ctx ctx = { ._prev = _grad_enabled };
    _grad_enabled = 0;
    return ctx;
}

void dnn_no_grad_exit(dnn_grad_ctx ctx) {
    _grad_enabled = ctx._prev;
}

/* ── grad_fn lifecycle ── */

grad_fn *_grad_fn_create(struct mem_pool *pool) {
    grad_fn *fn = _mem_pool_alloc(pool, sizeof(grad_fn), NULL);
    fn->pool = pool;
    return fn;
}

/* ── Gradient buffer management ── */

float *_grad_ensure(tensor *t) {
    tensor *base = t;
    while (base->parent) base = base->parent;
    if (!base->pool) {
        fprintf(stderr, "_grad_ensure: tensor %p has NULL pool (ndim=%d, shape=", (void*)base, base->ndim);
        for (int i = 0; i < base->ndim; i++) fprintf(stderr, "%s%d", i?",":"", base->shape[i]);
        fprintf(stderr, ", requires_grad=%d, grad_fn=%p)\n", base->requires_grad, (void*)base->grad_fn);
        assert(base->pool && "_grad_ensure: tensor has NULL pool");
    }
    if (!base->grad) {
        int n = tensor_numel(base);
        base->grad = _mem_pool_alloc(base->pool, n * sizeof(float), NULL);
    }
    return base->grad;
}

/* ── Backward ── */

typedef struct tensor_vec {
    tensor          **data;
    int              n;
    int              cap;
    struct mem_pool *pool;
} tensor_vec;

static int _vec_contains(const tensor_vec *v, tensor *t) {
    for (int i = 0; i < v->n; i++) if (v->data[i] == t) return 1;
    return 0;
}

static void _vec_push(tensor_vec *v, tensor *t) {
    if (v->n == v->cap) {
        int new_cap = v->cap ? v->cap * 2 : 256;
        tensor **new_data = _mem_pool_alloc(v->pool, (size_t)new_cap * sizeof(tensor*), NULL);
        if (v->data)
            memcpy(new_data, v->data, (size_t)v->n * sizeof(tensor*));
        v->data = new_data;
        v->cap = new_cap;
    }
    v->data[v->n++] = t;
}

static void _build_topo_from(tensor *t, tensor_vec *topo, tensor_vec *seen) {
    if (!t->grad_fn) return;
    if (_vec_contains(seen, t)) return;
    _vec_push(seen, t);

    for (int i = 0; i < t->grad_fn->n_inputs; i++)
        _build_topo_from(t->grad_fn->inputs[i], topo, seen);

    for (int i = 0; i < t->grad_fn->n_inputs; i++) {
        tensor *p = t->grad_fn->inputs[i];
        while (p->parent) {
            p = p->parent;
            _build_topo_from(p, topo, seen);
        }
    }

    _vec_push(topo, t);
}

void dnn_backward(struct mem_pool *scratch, tensor *loss) {
    assert(loss);

    tensor_vec topo = { .data = NULL, .n = 0, .cap = 0, .pool = scratch };
    tensor_vec seen = { .data = NULL, .n = 0, .cap = 0, .pool = scratch };
    _build_topo_from(loss, &topo, &seen);

    /* allocate and set loss gradient to all-ones */
    if (!loss->grad) {
        loss->grad = _mem_pool_alloc(loss->pool, tensor_numel(loss) * sizeof(float), NULL);
    }
    int numel = tensor_numel(loss);
    for (int i = 0; i < numel; i++) loss->grad[i] = 1.0f;

    /* reverse topological order */
    for (int i = topo.n - 1; i >= 0; i--) {
        tensor *t = topo.data[i];
        grad_fn *fn = t->grad_fn;
        if (!fn) continue;
        if (!t->grad) continue;

        tensor gv;
        memset(&gv, 0, sizeof(gv));
        gv.data   = (void*)t->grad;
        gv.ndim   = t->ndim;
        memcpy(gv.shape, t->shape, t->ndim * sizeof(int));
        int stride = 1;
        for (int d = t->ndim - 1; d >= 0; d--) {
            gv.strides[d] = stride;
            stride *= t->shape[d];
        }
        gv.offset = 0;
        gv.contiguous = 1;

        fn->backward(fn, &gv);
    }
}
