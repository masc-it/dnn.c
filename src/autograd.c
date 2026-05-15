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

static int _in_list(tensor *t, tensor **list, int n) {
    for (int i = 0; i < n; i++) if (list[i] == t) return 1;
    return 0;
}

/* Count reachable grad_fn nodes via DFS. Uses scratch-allocated seen array. */
static int _count_reachable(tensor *t, tensor **seen, int *n_seen) {
    if (!t->grad_fn) return 0;
    if (_in_list(t, seen, *n_seen)) return 0;
    seen[(*n_seen)++] = t;
    int count = 1;
    for (int i = 0; i < t->grad_fn->n_inputs; i++)
        count += _count_reachable(t->grad_fn->inputs[i], seen, n_seen);
    for (int i = 0; i < t->grad_fn->n_inputs; i++) {
        tensor *p = t->grad_fn->inputs[i];
        while (p->parent) {
            p = p->parent;
            if (p->grad_fn && !_in_list(p, seen, *n_seen))
                count += _count_reachable(p, seen, n_seen);
        }
    }
    return count;
}

static void _build_topo_from(tensor *t, tensor **topo, int *n,
                              tensor **seen, int *n_seen) {
    if (!t->grad_fn) return;
    if (_in_list(t, seen, *n_seen)) return;
    seen[(*n_seen)++] = t;

    for (int i = 0; i < t->grad_fn->n_inputs; i++)
        _build_topo_from(t->grad_fn->inputs[i], topo, n, seen, n_seen);

    for (int i = 0; i < t->grad_fn->n_inputs; i++) {
        tensor *p = t->grad_fn->inputs[i];
        while (p->parent) {
            p = p->parent;
            if (p->grad_fn && !_in_list(p, seen, *n_seen))
                _build_topo_from(p, topo, n, seen, n_seen);
        }
    }

    topo[(*n)++] = t;
}

void dnn_backward(struct mem_pool *scratch, tensor *loss) {
    assert(loss);

    /* First pass: count reachable grad_fn nodes */
    tensor **tmp = _mem_pool_alloc(scratch, 256 * sizeof(tensor*), NULL);
    int n_tmp = 0;
    int n_nodes = _count_reachable(loss, tmp, &n_tmp);

    tensor **topo = _mem_pool_alloc(scratch, n_nodes * sizeof(tensor*), NULL);
    tensor **seen = _mem_pool_alloc(scratch, 256 * sizeof(tensor*), NULL);
    int n_seen = 0, n = 0;
    _build_topo_from(loss, topo, &n, seen, &n_seen);
    assert(n == n_nodes && "dnn_backward: topo count mismatch");

    /* allocate and set loss gradient to all-ones */
    if (!loss->grad) {
        loss->grad = _mem_pool_alloc(loss->pool, tensor_numel(loss) * sizeof(float), NULL);
    }
    int numel = tensor_numel(loss);
    for (int i = 0; i < numel; i++) loss->grad[i] = 1.0f;

    /* reverse topological order */
    for (int i = n - 1; i >= 0; i--) {
        tensor *t = topo[i];
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
