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
/* ── Default pools ── */
void mem_pool_set_defaults(mem_pool *params, mem_pool *scratch, mem_pool *data);

/* ── Convenience (use default pools) ── */
void *mem_params_alloc(size_t bytes, const void *src);
void *mem_scratch_alloc(size_t bytes, const void *src);
void *mem_data_alloc(size_t bytes, const void *src);


#endif /* DNN_POOL_H */
