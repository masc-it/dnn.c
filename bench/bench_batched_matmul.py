# /// script
# requires-python = ">=3.10"
# dependencies = ["torch"]
# ///
"""PyTorch benchmark for batched matmul — mirrors bench_batched_matmul.c.

Usage:
    uv run bench_batched_matmul.py
"""

import torch
import time
import statistics

def bench(fn, warmup=3, trials=10):
    """Time fn, return median microseconds for fwd+bwd."""
    times = []
    for _ in range(warmup):
        fn()
    for _ in range(trials):
        t0 = time.perf_counter()
        fn()
        dt = time.perf_counter() - t0
        times.append(dt * 1e6)  # to us
    return statistics.median(times)


def fmt_gflops(us, flops):
    """flops / us * 1e3 = GFLOP/s"""
    return flops / (us * 1e3)


def run():
    print("=== PyTorch batched matmul benchmark ===")
    print(f"{'config':<28}  {'torch_fwd+bwd_us':>16}  {'gflop/s':>10}  {'batch_flops':>8}")
    print("-" * 70)

    # 2D reference
    for M, K, N in [(256, 256, 256), (512, 512, 512), (1024, 1024, 1024)]:
        def fn_2d(m=M, k=K, n=N):
            a = torch.randn(m, k, device='cpu', requires_grad=True)
            b = torch.randn(k, n, device='cpu', requires_grad=True)
            c = a @ b
            l = c.sum()
            l.backward()
        us = bench(fn_2d, warmup=3, trials=10)
        flops = 2.0 * M * K * N
        print(f"{'2D ref':<28}  {us:>16.0f}  {fmt_gflops(us, flops):>10.2f}  {'':>8}")

    print("-" * 70)

    # 3D batched
    cfgs = [
        (1,   64,   256,  256,   "B1_M64_K256_N256"),
        (1,   1,    512,  512,   "B1_M1_K512_N512"),
        (8,   512,  512,  512,   "B8_M512_K512_N512"),
        (8,   512,  512,  2048,  "B8_M512_K512_N2048"),
        (8,   512,  2048, 512,   "B8_M512_K2048_N512"),
        (32,  128,  256,  256,   "B32_M128_K256_N256"),
        (64,  64,   64,   64,    "B64_M64_K64_N64"),
        (2,   1,    4096, 4096,  "B2_M1_K4096_N4096"),
    ]

    for B, M, K, N, label in cfgs:
        def fn_3d(b=B, m=M, k=K, n=N):
            a = torch.randn(b, m, k, device='cpu', requires_grad=True)
            b = torch.randn(b, k, n, device='cpu', requires_grad=True)
            c = a @ b
            l = c.sum()
            l.backward()
        us = bench(fn_3d, warmup=3, trials=10)
        flops = 2.0 * B * M * K * N
        print(f"{label:<28}  {us:>16.0f}  {fmt_gflops(us, flops):>10.2f}  {flops/1e9:>7.1f}G")

    print("-" * 70)

    # Broadcast a
    print("--- Broadcast a: [M,K] @ [B,K,N] ---")
    for B, M, K, N in [(8, 512, 512, 512), (8, 512, 512, 2048), (32, 128, 256, 256)]:
        def fn_bcast_a(b=B, m=M, k=K, n=N):
            a = torch.randn(m, k, device='cpu', requires_grad=True)
            b = torch.randn(b, k, n, device='cpu', requires_grad=True)
            c = a @ b
            l = c.sum()
            l.backward()
        us = bench(fn_bcast_a, warmup=3, trials=10)
        flops = 2.0 * B * M * K * N
        print(f"bcast_a B={B:<2d} M={M:<3d} K={K:<3d} N={N:<3d}  {us:>10.0f}  {fmt_gflops(us, flops):>10.2f}  {flops/1e9:>7.1f}G")

    # 4D
    print("--- 4D: [B1,B2,M,K] @ [B1,B2,K,N] ---")
    for B1, B2, M, K, N in [(2, 4, 64, 64, 64), (2, 8, 128, 64, 64)]:
        def fn_4d(b1=B1, b2=B2, m=M, k=K, n=N):
            a = torch.randn(b1, b2, m, k, device='cpu', requires_grad=True)
            b = torch.randn(b1, b2, k, n, device='cpu', requires_grad=True)
            c = a @ b
            l = c.sum()
            l.backward()
        us = bench(fn_4d, warmup=3, trials=10)
        flops = 2.0 * B1 * B2 * M * K * N
        print(f"4d B1={B1} B2={B2} M={M} K={K} N={N}  {us:>10.0f}  {fmt_gflops(us, flops):>10.2f}  {flops/1e9:>7.1f}G")


if __name__ == "__main__":
    run()
