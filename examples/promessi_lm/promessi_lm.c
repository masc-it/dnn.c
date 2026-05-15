#include "promessi.h"

int main(void) {
    /* ── Pools ── */
    dnn_ctx ctx;
    dnn_ctx_init(&ctx, 128*1024*1024, (size_t)4*1024*1024*1024, 10*1024*1024);

    /* ── Load dataset ── */
    printf("Loading dataset...\n");
    lm_dataset ds = load_dataset("data/promessi.bin");
    if (!ds.data) {
        fprintf(stderr, "Failed to load data/promessi.bin\n");
        goto cleanup;
    }
    printf("  %d sequences, seq_len=%d, vocab=%d\n",
           ds.num_sequences, ds.seq_len, ds.vocab_size);
    printf("  total tokens: %ld\n", ds.data_n);

    /* verify vocab size matches */
    assert(ds.vocab_size == VOCAB_SIZE);

    /* ── Create model ── */
    printf("Creating decoder LM (%d layers, d_model=%d, n_heads=%d)...\n",
           N_LAYERS, D_MODEL, N_HEADS);

    /* set seed for reproducibility */
    srand(42);

    decoder_lm *lm = decoder_lm_create(ctx.params, VOCAB_SIZE, D_MODEL, N_LAYERS,
                                        N_HEADS, D_K, INTERMEDIATE);
    {
        long long n = decoder_lm_num_parameters(lm);
        printf("  model created.  Parameters: %lld (%.2fM)\n", n, n / 1e6);
        module_summary(&lm->base, 0, 0);
    }

    /* ── Init weights (GPT-2 style, overrides uniform default) ── */
    decoder_lm_init_weights(lm);
    printf("  Weights initialized: Normal(0,0.02), residual branches scaled by 1/sqrt(2*%d).\n", N_LAYERS);

    /* ── Enable RoPE ── */
    decoder_lm_enable_rope(ctx.params, lm, ds.seq_len, 10000.0f);
    printf("  RoPE enabled (base=10000.0, max_seq=%d).\n", ds.seq_len);

    /* ── Collect all trainable params ── */
    int n_params;
    tensor **all_params = module_parameters(&lm->base, &n_params);

    printf("  %d param groups.\n", n_params);

    /* ── Create optimizer ── */
    adamw_opt *opt = adamw_create(ctx.params, all_params, n_params, LR,
                                   0.9f, 0.999f, 1e-8f, 1e-4f);

    /* ── Training loop ── */
    int N          = ds.seq_len;
    int n_seqs     = ds.num_sequences;
    int batch_size = BATCH_SIZE;
    int n_batches  = (n_seqs + batch_size - 1) / batch_size;
#if OVERFIT
    if (n_batches > 10) n_batches = 10;
#endif

    /* ── Create LR scheduler (warmup + cosine) ── */
    int total_training_steps  = n_batches * MAX_EPOCHS;
    int warmup_steps          = n_batches;  /* warmup over 1 epoch */

    lr_scheduler *sched = lr_scheduler_create(ctx.params, opt, LR_SCHEDULE_LINEAR_WARMUP_COSINE,
                                                LR, warmup_steps, total_training_steps,
                                                MIN_LR,
                                                0, 0);
    printf("  LR scheduler: warmup=%d steps, cosine decay over %d steps, base_lr=%.2e, min_lr=%.2e\n",
           warmup_steps, total_training_steps, LR, MIN_LR);

    printf("\nTraining (AdamW, lr=%.4f, batch=%d, max_epochs=%d):\n",
           LR, batch_size, MAX_EPOCHS);
    printf("  sequences=%d  seq_len=%d  batches/epoch=%d\n",
           n_seqs, N, n_batches);

    /* shuffle index buffer */
    int *indices = malloc((size_t)n_seqs * sizeof(int));
    int total_steps = 0;

    for (int epoch = 0; epoch < MAX_EPOCHS; epoch++) {
        /* shuffle sequences (fixed subset in overfit mode) */
        for (int i = 0; i < n_seqs; i++) indices[i] = i;
#if !OVERFIT
        shuffle_int(indices, n_seqs);
#endif

        double epoch_loss = 0.0;
        int    epoch_batches = 0;
        struct timespec epoch_t0;
        clock_gettime(CLOCK_MONOTONIC, &epoch_t0);

        for (int b = 0; b < n_batches; b++) {
            int start = b * batch_size;
            int end   = start + batch_size;
            if (end > n_seqs) end = n_seqs;
            int bs    = end - start;

            /* ── Build batch tensor ── */
            tensor *input_ids = tensor_zeros_data(ctx.data, 2, (int[]){bs, N});
            int *id_data = (int *)input_ids->data;

            for (int i = 0; i < bs; i++) {
                int seq_idx = indices[start + i];
                int *src = ds.data + (long)seq_idx * N;
                memcpy(id_data + (long)i * N, src, (size_t)N * sizeof(int));
            }

            /* ── Train step ── */
            float grad_norm;
            tensor *loss = decoder_lm_train_step(ctx.scratch, ctx.data, lm, input_ids, opt, 1.0f, &grad_norm);
            float loss_val = ((float *)loss->data)[0];

            /* Advance LR scheduler after each training step */
            lr_scheduler_step(sched);

            epoch_loss    += loss_val;
            epoch_batches++;
            total_steps++;

            /* ── Log every LOG_EVERY batches ── */
            if ((b + 1) % LOG_EVERY == 0 || b == 0 || b == n_batches - 1) {
                struct timespec now;
                clock_gettime(CLOCK_MONOTONIC, &now);
                double elapsed = (double)(now.tv_sec - epoch_t0.tv_sec)
                               + (double)(now.tv_nsec - epoch_t0.tv_nsec) / 1e9;
                double batch_s = (double)(b + 1) / elapsed;
                float current_lr = lr_scheduler_get_lr(sched);

                printf("  epoch %2d/%d  batch %4d/%d  loss %.6f  lr %.2e  gn %.4e  %.1f batch/s\n",
                       epoch + 1, MAX_EPOCHS, b + 1, n_batches,
                       epoch_loss / epoch_batches, current_lr, grad_norm, batch_s);
            }

            /* ── Generate sample every GEN_EVERY batches ── */
            if ((b + 1) % GEN_EVERY == 0) {
                mem_pool_reset(ctx.scratch);

                int n_out;
                tensor *prompt = tensor_zeros_data(ctx.data, 2, (int[]){1, 1});
                ((int*)prompt->data)[0] = TOKENIZER_BOS_ID;  /* <|im_start|> */

                int *gen_ids = decoder_lm_generate(ctx.scratch, ctx.data, lm, prompt, GEN_NEW_TOKENS,
                                                    0.0f, 1, &n_out);

                tokenizer tok = tokenizer_with_chat_template();
                char *text = tokenizer_decode(&tok, gen_ids, n_out);
                printf("  ── gen (batch %d):\n  >> %s\n", b + 1, text);
                free(text);
            }

            /* free scratch + data */
            mem_pool_reset(ctx.scratch);
            mem_pool_reset(ctx.data);
        }

        struct timespec epoch_t1;
        clock_gettime(CLOCK_MONOTONIC, &epoch_t1);
        double epoch_sec = (double)(epoch_t1.tv_sec - epoch_t0.tv_sec)
                         + (double)(epoch_t1.tv_nsec - epoch_t0.tv_nsec) / 1e9;

        float epoch_end_lr = lr_scheduler_get_lr(sched);
        printf("  ── epoch %2d done  avg loss %.6f  lr %.2e  %.2fs  %.1f batch/s\n",
               epoch + 1, epoch_loss / epoch_batches,
               epoch_end_lr, epoch_sec, n_batches / epoch_sec);
    }

    printf("\nTraining complete.  Total steps: %d\n", total_steps);

    /* ── Final generation ── */
    {
        mem_pool_reset(ctx.scratch);
        mem_pool_reset(ctx.data);

        printf("\nFinal generation:\n");

        int n_out;
        tensor *prompt = tensor_zeros_data(ctx.data, 2, (int[]){1, 1});
        ((int*)prompt->data)[0] = TOKENIZER_BOS_ID;

        int *gen_ids = decoder_lm_generate(ctx.scratch, ctx.data, lm, prompt, GEN_NEW_TOKENS * 2,
                                            0.0f, 1, &n_out);

        tokenizer tok = tokenizer_with_chat_template();
        char *text = tokenizer_decode(&tok, gen_ids, n_out);
        printf("  >> %s\n", text);
        free(text);
    }

    /* ── Cleanup ── */
    free(indices);
    free(ds.data);
    adamw_free(opt);

cleanup:
    dnn_ctx_destroy(&ctx);

    printf("Done.\n");
    return ds.data ? 0 : 1;
}
