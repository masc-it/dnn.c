#include "autograd.h"
#include "_internal.h"
#include <assert.h>

/* ── grad_fn lifecycle ── */

grad_fn *grad_fn_create(void) {
    return mem_scratch_alloc(sizeof(grad_fn), NULL);
}

/* ── Backward ── */

void dnn_backward(tensor *loss) {
    (void)loss;
    /* TODO: topological sort + chain rule */
}
