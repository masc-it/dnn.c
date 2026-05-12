#include "pool.h"
#include <stdio.h>
#include <assert.h>

static void test_pool_alloc_copies_source_data(void) {
    mem_pool pool = mem_pool_create(256);
    mem_pool_set_defaults(&pool, NULL, NULL);
    float src[] = {1.0f, 2.0f, 3.0f};
    float *dst = mem_params_alloc(3 * sizeof(float), src);
    assert(dst[0] == 1.0f && dst[1] == 2.0f && dst[2] == 3.0f);
    mem_pool_destroy(&pool);
}

static void test_pool_reset_zeroes_next_allocation(void) {
    mem_pool pool = mem_pool_create(256);
    mem_pool_set_defaults(NULL, &pool, NULL);
    /* first alloc: write identifiable data */
    float *first = mem_scratch_alloc(4 * sizeof(float), NULL);
    first[0] = 1.0f; first[1] = 2.0f; first[2] = 3.0f; first[3] = 4.0f;
    /* reset the pool — rewinds offset, next alloc is at same position */
    mem_pool_reset(&pool);
    /* re-alloc: memset zeroes the reused memory */
    float *second = mem_scratch_alloc(4 * sizeof(float), NULL);
    assert(second[0] == 0.0f && second[1] == 0.0f);
    assert(second[2] == 0.0f && second[3] == 0.0f);
    mem_pool_destroy(&pool);
}

int main(void) {
    printf("test_pool: mem_pool lifecycle and alloc\n");
    test_pool_alloc_copies_source_data();
    printf("  alloc copies source: OK\n");
    test_pool_reset_zeroes_next_allocation();
    printf("  reset zeroes next alloc: OK\n");
    printf("    PASS\n");
    return 0;
}
