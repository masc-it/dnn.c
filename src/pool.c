#include "pool.h"
#include <stdlib.h>
#include <string.h>
#include <assert.h>

/* ── Lifecycle ── */

mem_pool mem_pool_create(size_t capacity) {
    mem_pool pool;
    pool.buffer  = malloc(capacity);
    assert(pool.buffer && "mem_pool_create: malloc failed");
    pool.capacity = capacity;
    pool.offset   = 0;
    return pool;
}

void mem_pool_destroy(mem_pool *pool) {
    if (pool && pool->buffer) {
        free(pool->buffer);
        pool->buffer  = NULL;
        pool->capacity = 0;
        pool->offset   = 0;
    }
}

void mem_pool_reset(mem_pool *pool) {
    pool->offset = 0;
}

size_t mem_pool_mark(mem_pool *pool) {
    return pool->offset;
}

void mem_pool_release(mem_pool *pool, size_t mark) {
    pool->offset = mark;
}

void *_mem_pool_alloc(mem_pool *pool, size_t bytes, const void *src) {
    assert(pool->offset + bytes <= pool->capacity
           && "_mem_pool_alloc: out of memory");
    void *ptr = pool->buffer + pool->offset;
    if (src) {
        memcpy(ptr, src, bytes);
    } else {
        memset(ptr, 0, bytes);
    }
    pool->offset += bytes;
    return ptr;
}

void *_mem_pool_alloc_nz(mem_pool *pool, size_t bytes) {
    assert(pool->offset + bytes <= pool->capacity
           && "_mem_pool_alloc_nz: out of memory");
    void *ptr = pool->buffer + pool->offset;
    pool->offset += bytes;
    return ptr;
}
