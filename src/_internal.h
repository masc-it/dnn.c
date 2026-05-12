#ifndef DNN_INTERNAL_H
#define DNN_INTERNAL_H

#include "pool.h"
#include "autograd.h"

/* ── Pool: low-level allocator ── */
void    *mem_pool_alloc(mem_pool *pool, size_t bytes, const void *src);

/* ── Pool: default pool accessors (internal) ── */
mem_pool *mem_pool_params(void);
mem_pool *mem_pool_scratch(void);
mem_pool *mem_pool_data(void);

/* ── Autograd: grad_fn factory (ops need this) ── */
grad_fn *grad_fn_create(void);

#endif /* DNN_INTERNAL_H */
