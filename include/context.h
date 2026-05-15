#ifndef DNN_CONTEXT_H
#define DNN_CONTEXT_H

#include "pool.h"

typedef struct dnn_ctx {
    mem_pool *params;   // weights, optimizer state, model structs — never reset
    mem_pool *scratch;  // activations, grad_fn, intermediates — reset per step
    mem_pool *data;     // raw data, batch tensors — reset per batch/epoch
} dnn_ctx;

// Convenience init — allocates all 3 pools from separate budgets.
// Returns 0 on success, -1 if any malloc fails.
int dnn_ctx_init(dnn_ctx *ctx, size_t params_sz, size_t scratch_sz, size_t data_sz);

// Reset scratch + data (called each step/batch). Params unchanged.
void dnn_ctx_reset_step(dnn_ctx *ctx);

// Destroy all 3 pools.
void dnn_ctx_destroy(dnn_ctx *ctx);

#endif
