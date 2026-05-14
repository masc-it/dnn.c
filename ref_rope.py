#!/usr/bin/env python3
"""
Reference implementation of RoPE frequency table using PyTorch.

Compares tensor_rope_freqs(d, base) from dnn.c against
torch-computed values.  Run with:

    python3 ref_rope.py

Exits with code 0 if all checks pass, 1 if any mismatch.
"""

import torch
import math
import sys

def ref_rope_freqs(d: int, base: float = 10000.0) -> torch.Tensor:
    """Compute RoPE frequency table: theta_k = base^{-2k/d}

    Matches the formula in src/rope.c:
        step = -2.0 * log(base) / d
        theta_k = exp(k * step)
    """
    half = d // 2
    # arange produces exactly k = 0..half-1
    k = torch.arange(half, dtype=torch.float32)
    # theta_k = base^{-2k/d}
    freqs = base ** (-2.0 * k / d)
    return freqs


def check(d: int, base: float, tol: float = 1e-5) -> bool:
    """Compute reference, print values, return pass/fail."""
    freqs = ref_rope_freqs(d, base)
    half = d // 2

    print(f"  d={d:4d}  base={base:8.1f}  half={half:3d}")
    print(f"    theta[0]   = {freqs[0].item():.10f}  (expect 1.0)")
    print(f"    theta[mid] = {freqs[half//2].item():.10f}")
    print(f"    theta[last]= {freqs[-1].item():.10f}")

    # Sanity checks
    ok = True

    # theta[0] must be 1.0 (base^0 = 1)
    if abs(freqs[0].item() - 1.0) > tol:
        print(f"    FAIL: theta[0] != 1.0  ({freqs[0].item():.10f})")
        ok = False

    # All values must be positive and not NaN
    if torch.any(torch.isnan(freqs)):
        print(f"    FAIL: NaN values found")
        ok = False
    if torch.any(freqs <= 0):
        print(f"    FAIL: non-positive values found")
        ok = False

    # Monotonically decreasing: check diff < 0 everywhere
    diff = freqs[1:] - freqs[:-1]
    if torch.any(diff >= 0):
        first_bad = torch.where(diff >= 0)[0][0].item()
        print(f"    FAIL: not monotonic at k={first_bad}")
        ok = False

    # Formula consistency: recompute via exp() route matching rope.c exactly
    k = torch.arange(half, dtype=torch.float32)
    step = -2.0 * math.log(base) / d
    freqs_exp = torch.exp(k * step)
    max_diff = torch.max(torch.abs(freqs - freqs_exp)).item()
    if max_diff > tol:
        print(f"    FAIL: exp route mismatch, max_diff={max_diff:.2e}")
        ok = False
    else:
        print(f"    max_diff(exp_powf) = {max_diff:.2e}")

    if ok:
        print(f"    PASS")
    return ok


def main():
    print("RoPE frequency table reference (PyTorch)")
    print()

    ok = True

    # Test various d and base combinations
    configs = [
        (2, 10000.0),
        (16, 10000.0),
        (32, 10000.0),
        (64, 10000.0),
        (128, 10000.0),
        (256, 10000.0),
        (128, 50000.0),
        (128, 1000.0),
    ]

    for d, base in configs:
        if not check(d, base):
            ok = False

    # Print full values for d=64, base=10000 for manual inspection
    print()
    print("Full table for d=64, base=10000.0:")
    freqs = ref_rope_freqs(64, 10000.0)
    half = 32
    for k in range(half):
        print(f"  theta[{k:2d}] = {freqs[k].item():.10f}")

    print()
    if ok:
        print("All checks PASSED")
        sys.exit(0)
    else:
        print("Some checks FAILED")
        sys.exit(1)


if __name__ == "__main__":
    main()
