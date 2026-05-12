#include "ops.h"
#include "autograd.h"
#include "pool.h"
#include "tensor_int.h"
#include "autograd_int.h"
#include "broadcast.h"
#include "ops_activation_int.h"
#include <stdlib.h>
#include <assert.h>
#include <string.h>

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
