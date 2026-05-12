#include "dnn.h"
#include <stdio.h>

int main(void) {
    mem_pool params  = mem_pool_create(64 * 1024);
    mem_pool scratch = mem_pool_create(64 * 1024);
    mem_pool_set_defaults(&params, &scratch, NULL);

    tensor *t = tensor_randn(2, (int[]){2, 3}, 0);
    tensor_print(t);

    /* flatten via -1 inference */
    tensor *v = tensor_flatten(t);
    tensor_print(v);

    /* reshape with -1 auto-inference: [3, -1] -> [3, 2] */
    tensor *w = tensor_reshape(t, 2, (int[]){3, -1});
    tensor_print(w);

    mem_pool_destroy(&params);
    mem_pool_destroy(&scratch);
    return 0;
}
