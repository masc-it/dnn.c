#ifndef DNN_CONV_H
#define DNN_CONV_H

#include "tensor.h"

/* ── Conv2D ──
 *
 *   y = x ✶ weight + bias   (cross-correlation, not convolution — matches PyTorch)
 *
 *   input  — shape (N, C, H, W), MUST be contiguous
 *   weight — shape (out_C, in_C, kH, kW), MUST be contiguous
 *   bias   — shape (out_C,), may be NULL (treated as 0)
 *   stride — spatial stride (same for H and W)
 *   pad    — zero-padding on both sides of H and W
 *
 *   output shape: (N, out_C, H_out, W_out) where
 *     H_out = (H + 2*pad - kH) / stride + 1
 *     W_out = (W + 2*pad - kW) / stride + 1
 */
tensor *tensor_conv2d(struct mem_pool *scratch,
                      tensor *input, tensor *weight, tensor *bias,
                      int stride, int pad);

#endif /* DNN_CONV_H */
