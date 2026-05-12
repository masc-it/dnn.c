#include "ops.h"
#include "autograd.h"
#include "pool.h"
#include "tensor_int.h"
#include "autograd_int.h"
#include "broadcast.h"
#include "ops_matrix_int.h"
#include <stdlib.h>
#include <assert.h>
#include <string.h>

/* ── Matrix ops ── */

tensor *tensor_matmul(const tensor *a, const tensor *b) {
    (void)a; (void)b;
    return NULL;
}
