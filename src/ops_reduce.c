#include "ops.h"
#include "autograd.h"
#include "pool.h"
#include "tensor_int.h"
#include "autograd_int.h"
#include "broadcast.h"
#include "ops_reduce_int.h"
#include <stdlib.h>
#include <assert.h>
#include <string.h>

/* ── Reduction ── */

tensor *tensor_sum(const tensor *t, int dim) {
    (void)t; (void)dim;
    return NULL;
}

tensor *tensor_mean(const tensor *t, int dim) {
    (void)t; (void)dim;
    return NULL;
}
