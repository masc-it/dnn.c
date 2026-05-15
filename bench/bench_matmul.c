#include "dnn.h"
#include "context.h"
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

static dnn_ctx ctx;

static double now_us(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec*1e6 + (double)ts.tv_nsec/1e3;
}

static double time_matmul(int M, int K, int N, int warmup, int trials) {
    double *t = malloc(trials * sizeof(double));
    for (int tr = -warmup; tr < trials; tr++) {
        mem_pool p = mem_pool_create(64*1024*1024);
        mem_pool s = mem_pool_create(64*1024*1024);
        mem_pool_set_defaults(&p, &s, NULL);
        tensor *a = tensor_randn(ctx.params, 2, (int[]){M, K}, 1);
        tensor *b = tensor_randn(ctx.params, 2, (int[]){K, N}, 1);
        double t0 = now_us();
        tensor *c = tensor_matmul(ctx.scratch, a, b);
        tensor *l = tensor_sum(ctx.scratch, c, 0);
        dnn_backward(ctx.scratch, l);
        double dt = now_us() - t0;
        dnn_ctx_destroy(&ctx);
    // mem_pool_destroy(&p);
        dnn_ctx_destroy(&ctx);
    // mem_pool_destroy(&s);
        if (tr >= 0) t[tr] = dt;
    }
    for (int i=0;i<trials;i++) for(int j=i+1;j<trials;j++) if(t[i]>t[j]){double tmp=t[i];t[i]=t[j];t[j]=tmp;}
    double med = trials%2 ? t[trials/2] : (t[trials/2-1]+t[trials/2])/2.0;
    free(t);
    return med;
}

int main(void) {
    int cfgs[][3] = {{256,128,64},{512,256,128},{1024,512,256},{128,1024,128},{4096,144,32},{64,256,256},{1024,1024,1024}};
    int n = sizeof(cfgs)/sizeof(cfgs[0]);
    printf("# M K N   median_us  GFLOP/s\n");
    for (int i=0;i<n;i++) {
        int M=cfgs[i][0],K=cfgs[i][1],N=cfgs[i][2];
        double us = time_matmul(M,K,N,2,5);
        double flops = 2.0*M*K*N;
        printf("%d %d %d   %8.0f   %.2f\n", M,K,N,us,flops/(us*1e3));
    }
    return 0;
}
