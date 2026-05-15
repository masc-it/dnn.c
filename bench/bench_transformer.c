#include "dnn.h"
#include "context.h"
#include "transformer.h"
#include "optim.h"
#include "pool.h"
#include <stdio.h>
#include <time.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

static dnn_ctx ctx;

/* ── Decoder LM benchmark at realistic scale ──
 *
 * Measures fwd+bwd timing per step for a moderate-sized decoder LM.
 * d_model=256, n_layers=2, n_heads=4, d_k=64, intermediate=768
 * Sequence length N=128, batch B=2, vocab=32000.
 *
 * Prints per-step timing and estimated throughput.
 */

static tensor *make_int_tensor(int ndim, const int *shape) {
    int n = 1;
    for (int i = 0; i < ndim; i++) n *= shape[i];
    tensor *t = tensor_zeros_data(ctx.data, ndim, shape);
    for (int i = 0; i < n; i++) ((int*)t->data)[i] = rand() % 100;
    return t;
}

int main(void) {
    printf("═══ Transformer benchmark ═══\n\n");

    /* ── Configuration ── */
    int B        = 2;
    int N        = 128;
    int vocab    = 32000;
    int d_model  = 256;
    int n_layers = 2;
    int n_heads  = 4;
    int d_k      = 64;
    int intermediate = 768;
    int warmup   = 3;
    int iters    = 10;

    printf("Config:\n");
    printf("  B=%d, N=%d, vocab=%d\n", B, N, vocab);
    printf("  d_model=%d, n_layers=%d, n_heads=%d, d_k=%d\n",
           d_model, n_layers, n_heads, d_k);
    printf("  intermediate=%d\n", intermediate);
    printf("\n");

    /* Estimate param count for pool size */
    size_t param_pool_sz = 512 * 1024 * 1024;
    size_t scratch_pool_sz = 512 * 1024 * 1024;  /* activations for B=2, N=128 */
    size_t data_pool_sz = 16 * 1024 * 1024;

    dnn_ctx_init(&ctx, 8*1024*1024, 64*1024*1024, 8*1024*1024);

    srand(42);
    /* grad mode on by default */

    /* ── Create model ── */
    decoder_lm *lm = decoder_lm_create(ctx.params, vocab, d_model, n_layers, n_heads,
                                        d_k, intermediate);

    printf("  Parameters: %.1fM\n", (double)decoder_lm_num_parameters(lm) / 1e6);

    /* Collect params for optimizer */
    int n_params = 0;
    tensor *all_params[256];
    all_params[n_params++] = lm->embedding_table;
    all_params[n_params++] = lm->norm_weight;
    all_params[n_params++] = lm->norm_bias;
    /* lm_head->weight excluded — weight tying via transposed view of embedding_table */
    all_params[n_params++] = lm->lm_head->bias;
    for (int i = 0; i < n_layers; i++) {
        transformer_block *b = lm->blocks[i];
        all_params[n_params++] = b->q_proj->weight;
        all_params[n_params++] = b->q_proj->bias;
        all_params[n_params++] = b->k_proj->weight;
        all_params[n_params++] = b->k_proj->bias;
        all_params[n_params++] = b->v_proj->weight;
        all_params[n_params++] = b->v_proj->bias;
        all_params[n_params++] = b->out_proj->weight;
        all_params[n_params++] = b->out_proj->bias;
        all_params[n_params++] = b->attn_norm_weight;
        all_params[n_params++] = b->attn_norm_bias;
        all_params[n_params++] = b->ffn_norm_weight;
        all_params[n_params++] = b->ffn_norm_bias;
        all_params[n_params++] = b->ffn->gate_proj->weight;
        all_params[n_params++] = b->ffn->gate_proj->bias;
        all_params[n_params++] = b->ffn->up_proj->weight;
        all_params[n_params++] = b->ffn->up_proj->bias;
        all_params[n_params++] = b->ffn->down_proj->weight;
        all_params[n_params++] = b->ffn->down_proj->bias;
    }

    tensor **pz = _mem_pool_alloc(ctx.params, n_params * sizeof(tensor*), NULL);
    memcpy(pz, all_params, n_params * sizeof(tensor*));
    adamw_opt *opt = adamw_create(ctx.params, pz, n_params, 0.001f, 0.9f, 0.999f, 1e-8f, 0.01f);

    /* ── Input data ── */
    tensor *input_ids = make_int_tensor(2, (int[]){B, N});

    printf("Starting benchmark (%d warmup + %d measured iterations)...\n\n", warmup, iters);

    /* ── Warmup ── */
    for (int i = 0; i < warmup; i++) {
        tensor *loss = decoder_lm_train_step(ctx.scratch, ctx.data, lm, input_ids, opt, 1.0f, NULL);
        (void)loss;
        mem_pool_reset(ctx.scratch);
        mem_pool_reset(ctx.data);
        input_ids = make_int_tensor(2, (int[]){B, N});
    }

    /* ── Measured iterations ── */
    double total_time = 0.0;
    double min_time = 1e9, max_time = 0.0;

    for (int i = 0; i < iters; i++) {
        clock_t start = clock();

        tensor *loss = decoder_lm_train_step(ctx.scratch, ctx.data, lm, input_ids, opt, 1.0f, NULL);

        clock_t end = clock();
        double elapsed = (double)(end - start) / CLOCKS_PER_SEC;

        total_time += elapsed;
        if (elapsed < min_time) min_time = elapsed;
        if (elapsed > max_time) max_time = elapsed;

        float loss_val = ((float*)loss->data)[0];
        printf("  iter %2d: %7.1f ms  (loss %.4f)\n", i + 1, elapsed * 1000, loss_val);

        mem_pool_reset(ctx.scratch);
        mem_pool_reset(ctx.data);
        input_ids = make_int_tensor(2, (int[]){B, N});
    }

    double avg = total_time / iters;
    double throughput = 1.0 / avg;
    double tokens_per_sec = (double)B * N * throughput;

    printf("\n═══ Results ═══\n");
    printf("  Avg step time:  %7.1f ms\n", avg * 1000);
    printf("  Min step time:  %7.1f ms\n", min_time * 1000);
    printf("  Max step time:  %7.1f ms\n", max_time * 1000);
    printf("  Steps/sec:      %7.1f\n", throughput);
    printf("  Tokens/sec:     %7.0f\n", tokens_per_sec);
    printf("  (B=%d, N=%d, %d tokens/step)\n", B, N, B * N);
    printf("\n");

    printf("Benchmark complete.\n");
    dnn_ctx_destroy(&ctx);

    return 0;
}
