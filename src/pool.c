#include "pool.h"
#include <stdlib.h>
#include <string.h>
#include <assert.h>

/* ── Default pool globals ── */

static mem_pool *pool_params  = NULL;
static mem_pool *pool_scratch = NULL;
static mem_pool *pool_data    = NULL;

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

void *mem_pool_alloc(mem_pool *pool, size_t bytes, const void *src) {
    assert(pool->offset + bytes <= pool->capacity
           && "mem_pool_alloc: out of memory");
    void *ptr = pool->buffer + pool->offset;
    if (src) {
        memcpy(ptr, src, bytes);
    } else {
        memset(ptr, 0, bytes);
    }
    pool->offset += bytes;
    return ptr;
}

/* ── Default pools ── */

void mem_pool_set_defaults(mem_pool *params, mem_pool *scratch, mem_pool *data) {
    pool_params  = params;
    pool_scratch = scratch;
    pool_data    = data;
}

void *mem_params_alloc(size_t bytes, const void *src) {
    assert(pool_params && "mem_params_alloc: no default params pool set");
    return mem_pool_alloc(pool_params, bytes, src);
}

void *mem_scratch_alloc(size_t bytes, const void *src) {
    assert(pool_scratch && "mem_scratch_alloc: no default scratch pool set");
    return mem_pool_alloc(pool_scratch, bytes, src);
}

void *mem_data_alloc(size_t bytes, const void *src) {
    assert(pool_data && "mem_data_alloc: no default data pool set");
    return mem_pool_alloc(pool_data, bytes, src);
}

mem_pool *mem_pool_params(void) {
    return pool_params;
}

mem_pool *mem_pool_scratch(void) {
    return pool_scratch;
}

mem_pool *mem_pool_data(void) {
    return pool_data;
}
