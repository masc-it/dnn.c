#include "dnn.h"
#include "nn.h"
#include <stdio.h>
#include <string.h>

static void zero_grad(tensor *t) {
    float *g = tensor_grad(t);
    if (g) memset(g, 0, tensor_numel(t) * sizeof(float));
}

void mlp_xor(mem_pool *scratch) {
    printf("── XOR MLP ──\n");

    /* data: 4 samples, 2 features */
    tensor *X = tensor_zeros(2, (int[]){4, 2}, 0);
    float *xp = tensor_data_ptr(X);
    xp[0]=0; xp[1]=0;
    xp[2]=0; xp[3]=1;
    xp[4]=1; xp[5]=0;
    xp[6]=1; xp[7]=1;

    /* targets: class indices (0 or 1), stored as ints in float buffer */
    tensor *y = tensor_zeros(1, (int[]){4}, 0);
    int *yp = (int*)tensor_data_ptr(y);
    yp[0]=0; yp[1]=1; yp[2]=1; yp[3]=0;

    /* model: 2 → 16 → 2 */
    linear *l1 = linear_create(2, 16);
    linear *l2 = linear_create(16, 2);
    linear *layers[] = {l1, l2};
    int n_layers = 2;

    float lr = 0.5f;
    int epochs = 2000;

    for (int epoch = 0; epoch < epochs; epoch++) {
        mem_pool_reset(scratch);

        /* forward */
        tensor *h = linear_forward(l1, X);
        tensor *r = tensor_relu(h);
        tensor *logits = linear_forward(l2, r);
        tensor *loss = tensor_cross_entropy(logits, y, 1);

        /* backward */
        dnn_backward(loss);

        /* SGD: param -= lr * grad */
        for (int li = 0; li < n_layers; li++) {
            float *wd = tensor_data_ptr(layers[li]->weight);
            float *wg = tensor_grad(layers[li]->weight);
            int wn = tensor_numel(layers[li]->weight);
            for (int j = 0; j < wn; j++) wd[j] -= lr * wg[j];

            float *bd = tensor_data_ptr(layers[li]->bias);
            float *bg = tensor_grad(layers[li]->bias);
            int bn = tensor_numel(layers[li]->bias);
            for (int j = 0; j < bn; j++) bd[j] -= lr * bg[j];
        }

        /* zero grads for next iteration */
        for (int li = 0; li < n_layers; li++) {
            zero_grad(layers[li]->weight);
            zero_grad(layers[li]->bias);
        }

        if (epoch % 500 == 0 || epoch == epochs - 1) {
            float loss_val = ((float*)loss->data)[0];
            printf("  epoch %4d, loss %.6f\n", epoch, loss_val);
        }
    }

    /* evaluate */
    printf("\n  predictions:\n");
    dnn_grad_ctx ctx = dnn_no_grad_enter();
    tensor *h = linear_forward(l1, X);
    tensor *r = tensor_relu(h);
    tensor *logits = linear_forward(l2, r);
    dnn_no_grad_exit(ctx);

    float *ld = tensor_data_ptr(logits);
    for (int i = 0; i < 4; i++) {
        int pred = ld[i*2 + 1] > ld[i*2] ? 1 : 0;
        int correct = pred == yp[i];
        printf("    [%d %d] → %d  (true %d)%s\n",
               (int)xp[i*2], (int)xp[i*2+1], pred, yp[i],
               correct ? "" : " ✗");
    }
    printf("  done.\n");
}

int main(void) {
    mem_pool params  = mem_pool_create(512 * 1024);
    mem_pool scratch = mem_pool_create(512 * 1024);
    mem_pool_set_defaults(&params, &scratch, NULL);

    mlp_xor(&scratch);

    mem_pool_destroy(&params);
    mem_pool_destroy(&scratch);
    return 0;
}
