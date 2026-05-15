#ifndef DNN_POOL_H
#define DNN_POOL_H

#include <stddef.h>

typedef struct mem_pool {
    char   *buffer;
    size_t  capacity;
    size_t  offset;
} mem_pool;

/* ── Lifecycle ── */
mem_pool mem_pool_create(size_t capacity);
void     mem_pool_destroy(mem_pool *pool);
void     mem_pool_reset(mem_pool *pool);
size_t   mem_pool_mark(mem_pool *pool);
void     mem_pool_release(mem_pool *pool, size_t mark);

/* ── Low-level allocators (exposed for tensor.h etc.) ── */
void    *_mem_pool_alloc(struct mem_pool *pool, size_t bytes, const void *src);
void    *_mem_pool_alloc_nz(struct mem_pool *pool, size_t bytes);

#endif /* DNN_POOL_H */
