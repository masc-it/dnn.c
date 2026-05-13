#ifndef DNN_POOL_INT_H
#define DNN_POOL_INT_H

#include "pool.h"

/* ── Low-level allocators (exposed for tensor.c) ── */
void    *_mem_pool_alloc(mem_pool *pool, size_t bytes, const void *src);
void    *_mem_pool_alloc_nz(mem_pool *pool, size_t bytes);

/* ── Default pool accessors (exposed for tensor.c) ── */
mem_pool *_mem_pool_params(void);
mem_pool *_mem_pool_scratch(void);
mem_pool *_mem_pool_data(void);

#endif /* DNN_POOL_INT_H */
