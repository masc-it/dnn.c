#include "context.h"
#include <stdlib.h>
#include <assert.h>

int dnn_ctx_init(dnn_ctx *ctx, size_t params_sz, size_t scratch_sz, size_t data_sz) {
    mem_pool *p = malloc(sizeof(mem_pool));
    mem_pool *s = malloc(sizeof(mem_pool));
    mem_pool *d = malloc(sizeof(mem_pool));
    if (!p || !s || !d) { free(p); free(s); free(d); return -1; }
    *p = mem_pool_create(params_sz);
    *s = mem_pool_create(scratch_sz);
    *d = mem_pool_create(data_sz);
    if (!p->buffer || !s->buffer || !d->buffer) {
        mem_pool_destroy(p); free(p);
        mem_pool_destroy(s); free(s);
        mem_pool_destroy(d); free(d);
        return -1;
    }
    ctx->params  = p;
    ctx->scratch = s;
    ctx->data    = d;
    return 0;
}

void dnn_ctx_reset_step(dnn_ctx *ctx) {
    mem_pool_reset(ctx->scratch);
    mem_pool_reset(ctx->data);
}

void dnn_ctx_destroy(dnn_ctx *ctx) {
    if (ctx->params)  { mem_pool_destroy(ctx->params);  free(ctx->params);  }
    if (ctx->scratch) { mem_pool_destroy(ctx->scratch); free(ctx->scratch); }
    if (ctx->data)    { mem_pool_destroy(ctx->data);    free(ctx->data);    }
    ctx->params = ctx->scratch = ctx->data = NULL;
}
