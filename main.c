#include "dnn.h"
#include "nn.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static void train_xor(const char *label, mem_pool *scratch,
                      void (*step_fn)(void*),
                      void (*zero_grad_fn)(void*),
                      void *opt, linear *l1, linear *l2) {
    printf("── XOR MLP (%s, seed=42) ──\n", label);

    srand(42);

    tensor *X = tensor_zeros(2, (int[]){4, 2}, 0);
    float *xp = tensor_data_ptr(X);
    xp[0]=0; xp[1]=0; xp[2]=0; xp[3]=1; xp[4]=1; xp[5]=0; xp[6]=1; xp[7]=1;

    tensor *y = tensor_zeros(1, (int[]){4}, 0);
    int *yp = (int*)tensor_data_ptr(y);
    yp[0]=0; yp[1]=1; yp[2]=1; yp[3]=0;

    int epochs = 200;
    for (int epoch = 0; epoch < epochs; epoch++) {
        mem_pool_reset(scratch);

        tensor *h  = linear_forward(l1, X);
        tensor *r  = tensor_relu(h);
        tensor *logits = linear_forward(l2, r);
        tensor *loss   = tensor_cross_entropy(logits, y, 1);

        dnn_backward(loss);

        step_fn(opt);
        zero_grad_fn(opt);

        if (epoch % 40 == 0 || epoch == epochs - 1) {
            printf("  epoch %3d, loss %.6f\n", epoch, ((float*)loss->data)[0]);
        }
    }

    printf("\n  predictions:\n");
    dnn_grad_ctx ctx = dnn_no_grad_enter();
    tensor *h = linear_forward(l1, X);
    tensor *r = tensor_relu(h);
    tensor *logits = linear_forward(l2, r);
    dnn_no_grad_exit(ctx);

    float *ld = tensor_data_ptr(logits);
    for (int i = 0; i < 4; i++) {
        int pred = ld[i*2 + 1] > ld[i*2] ? 1 : 0;
        printf("    [%d %d] → %d  (true %d)%s\n",
               (int)xp[i*2], (int)xp[i*2+1], pred, yp[i],
               pred == yp[i] ? "" : " ✗");
    }
    printf("  done.\n\n");
}

static void sgd_step_wrap(void *p) { sgd_step((sgd_opt*)p); }
static void sgd_zg_wrap(void *p)   { sgd_zero_grad((sgd_opt*)p); }
static void adamw_step_wrap(void *p) { adamw_step((adamw_opt*)p); }
static void adamw_zg_wrap(void *p)   { adamw_zero_grad((adamw_opt*)p); }

int main(void) {
    mem_pool params  = mem_pool_create(1024 * 1024);
    mem_pool scratch = mem_pool_create(1024 * 1024);
    mem_pool_set_defaults(&params, &scratch, NULL);

    /* SGD */
    srand(42);
    linear *l1 = linear_create(2, 16);
    linear *l2 = linear_create(16, 2);
    tensor *sgd_params[] = {l1->weight, l1->bias, l2->weight, l2->bias};
    sgd_opt *sgd = sgd_create(sgd_params, 4, 0.1f, 0.9f);

    train_xor("SGD, lr=0.1, momentum=0.9", &scratch,
              sgd_step_wrap, sgd_zg_wrap, sgd, l1, l2);

    /* AdamW with fresh init */
    srand(42);
    linear *l1b = linear_create(2, 16);
    linear *l2b = linear_create(16, 2);
    tensor *aw_params[] = {l1b->weight, l1b->bias, l2b->weight, l2b->bias};
    adamw_opt *adamw = adamw_create(aw_params, 4, 0.01f, 0.9f, 0.999f, 1e-8f, 0.01f);

    train_xor("AdamW, lr=0.01, wd=0.01", &scratch,
              adamw_step_wrap, adamw_zg_wrap, adamw, l1b, l2b);

    mem_pool_destroy(&params);
    mem_pool_destroy(&scratch);
    return 0;
}
