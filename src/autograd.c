#include "autograd.h"
#include "pool.h"
#include <stdlib.h>
#include <string.h>
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

grad_fn *_grad_fn_create(void) {
    return mem_scratch_alloc(sizeof(grad_fn), NULL);
}

/* ── Gradient buffer management ── */

float *_grad_ensure(tensor *t) {
    tensor *base = t;
    while (base->parent) base = base->parent;
    if (!base->grad) {
        int n = tensor_numel(base);
        /* leaf params: persistent grads.  intermediates: ephemeral, reclaimed on scratch reset */
        if (base->requires_grad && !base->grad_fn)
            base->grad = mem_params_alloc(n * sizeof(float), NULL);
        else
            base->grad = mem_scratch_alloc(n * sizeof(float), NULL);
    }
    return base->grad;
}

/* ── Backward ── */

static int _in_list(tensor *t, tensor **list, int n) {
    for (int i = 0; i < n; i++) if (list[i] == t) return 1;
    return 0;
}

/* DFS topological sort: collect non-leaf tensors in dependency order */
static void _build_topo(tensor *t, tensor ***topo, int *n, int *cap) {
    if (!t->grad_fn) return;
    if (_in_list(t, *topo, *n)) return;

    for (int i = 0; i < t->grad_fn->n_inputs; i++) {
        _build_topo(t->grad_fn->inputs[i], topo, n, cap);
    }

    /* walk parent chains of each input — views have no grad_fn but their
       parent tensors may, and those need to be in the topo order for
       gradient to continue flowing backward. */
    for (int i = 0; i < t->grad_fn->n_inputs; i++) {
        tensor *p = t->grad_fn->inputs[i];
        while (p->parent) {
            p = p->parent;
            if (p->grad_fn && !_in_list(p, *topo, *n))
                _build_topo(p, topo, n, cap);
        }
    }

    if (*n >= *cap) {
        *cap = *cap ? *cap * 2 : 64;
        *topo = realloc(*topo, *cap * sizeof(tensor*));
        assert(*topo && "_build_topo: realloc failed");
    }
    (*topo)[(*n)++] = t;
}

void dnn_backward(tensor *loss) {
    assert(loss);

    /* build topological order of non-leaf tensors reachable from loss */
    int cap = 0, n = 0;
    tensor **topo = NULL;
    _build_topo(loss, &topo, &n, &cap);

    /* allocate and set loss gradient to all-ones (grad of sum(loss)) */
    if (!loss->grad) {
        if (loss->requires_grad && !loss->grad_fn)
            loss->grad = mem_params_alloc(tensor_numel(loss) * sizeof(float), NULL);
        else
            loss->grad = mem_scratch_alloc(tensor_numel(loss) * sizeof(float), NULL);
    }
    int numel = tensor_numel(loss);
    for (int i = 0; i < numel; i++) loss->grad[i] = 1.0f;

    /* reverse topological order: call each grad_fn with its output gradient */
    for (int i = n - 1; i >= 0; i--) {
        tensor *t = topo[i];
        grad_fn *fn = t->grad_fn;
        if (!fn) continue;
        if (!t->grad) continue;  /* no gradient flowed to this node */

        /* wrap t->grad in a minimal tensor so backward fn can read shape/strides */
        tensor gv;
        memset(&gv, 0, sizeof(gv));
        gv.data   = (void*)t->grad;
        gv.ndim   = t->ndim;
        memcpy(gv.shape, t->shape, t->ndim * sizeof(int));
        /* contiguous strides so _flat_off reads grad elements in order */
        int stride = 1;
        for (int d = t->ndim - 1; d >= 0; d--) {
            gv.strides[d] = stride;
            stride *= t->shape[d];
        }
        gv.offset = 0;

        fn->backward(fn, &gv);
    }

    free(topo);
}
