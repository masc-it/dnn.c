#include "dnn.h"
#include "gpt.h"
#include "vlm.h"
#include "context.h"
#include "optim.h"
#include <stdio.h>
#include <assert.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>

static dnn_ctx ctx;

int main(void) {
    printf("=== VLM Bench Debug ===\n");
    int I = 16, T = 32, d_model = 128, n_layers = 2, n_heads = 4, d_k = 32, intermediate = 256;
    int C = 3, patch = 16;
    int pp = 4;
    int H_ = pp * patch, W_ = pp * patch;

    dnn_ctx_init(&ctx, 64*1024*1024, 512*1024*1024, 32*1024*1024);
    dnn_seed(0);

    printf("Creating VLM...\n");
    vision_lm *vlm = vision_lm_create(ctx.params, 261, d_model, n_layers,
                                       n_heads, d_k, intermediate,
                                       C, H_, W_, patch, 1);
    printf("  image_channels=%d image_h=%d image_w=%d\n",
           vlm->image_channels, vlm->image_h, vlm->image_w);
    printf("  patch_embed->in_channels=%d out_channels=%d kernel=%d stride=%d\n",
           vlm->patch_embed->in_channels, vlm->patch_embed->out_channels,
           vlm->patch_embed->kernel_size, vlm->patch_embed->stride);
    printf("  patch_embed->weight shape: ");
    for (int i = 0; i < vlm->patch_embed->weight->ndim; i++)
        printf("%d ", vlm->patch_embed->weight->shape[i]);
    printf("\n");

    vision_lm_init_weights(vlm);
    printf("After init_weights, image_channels=%d\n", vlm->image_channels);

    printf("Creating image tensor...\n");
    tensor *images = tensor_randn(ctx.scratch, 4, (int[]){1, C, H_, W_}, 0);
    printf("  images shape: %d %d %d %d contiguous=%d\n",
           images->shape[0], images->shape[1], images->shape[2], images->shape[3],
           tensor_is_contiguous(images));
    printf("  vlm->image_channels=%d images->shape[1]=%d\n",
           vlm->image_channels, images->shape[1]);

    printf("Calling vision_lm_image_embeds...\n");
    tensor *e = vision_lm_image_embeds(ctx.scratch, vlm, images);
    printf("  embeds shape: %d %d %d\n", e->shape[0], e->shape[1], e->shape[2]);
    printf("OK\n");
    return 0;
}
