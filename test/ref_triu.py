#!/usr/bin/env python3
"""Reference: compare tensor_triu output against PyTorch triu-based causal mask.

Usage:
    python3 ref_triu.py                     # run all checks
    python3 ref_triu.py --print-table       # print expected tables as C test data

The dnn.c tensor_triu(N, diagonal) creates [N, N] float mask:
    (i,j) = -inf  if j >= i + diagonal
    (i,j) =  0    otherwise

This matches PyTorch:
    mask = torch.full((N, N), float('-inf'))
    mask = torch.triu(mask, diagonal=diagonal)
    # triu with diag=d: keep elements where j >= i + d as -inf,
    #                   zero out elements where j < i + d.
"""

import torch
import math
import sys

def triu_mask_pytorch(N, diagonal):
    """Produce same result as dnn.c tensor_triu(N, diagonal)."""
    mask = torch.full((N, N), float('-inf'))
    mask = torch.triu(mask, diagonal=diagonal)
    return mask

def triu_mask_dnn(N, diagonal):
    """Emulate dnn.c tensor_triu formula."""
    mask = torch.zeros(N, N)
    for i in range(N):
        for j in range(N):
            if j >= i + diagonal:
                mask[i, j] = float('-inf')
    return mask

def check(N, diagonal):
    pt = triu_mask_pytorch(N, diagonal)
    dnn = triu_mask_dnn(N, diagonal)
    match = torch.equal(pt, dnn)
    if not match:
        print(f"  MISMATCH N={N} diag={diagonal}")
        print("  PyTorch triu result:")
        print(pt)
        print("  dnn.c formula result:")
        print(dnn)
    return match

def print_table(N, diagonal):
    """Print expected table as C test data."""
    mask = triu_mask_pytorch(N, diagonal)
    print(f"// N={N}, diagonal={diagonal}")
    for i in range(N):
        row = []
        for j in range(N):
            if mask[i, j] == 0:
                row.append(f"0")
            else:
                row.append(f"-inf")
        print(f"//   {i}: " + " ".join(f"{v:>5}" for v in row))
    print(f"// Flat expected[] array:")
    arr = []
    for i in range(N):
        for j in range(N):
            arr.append("0" if mask[i, j] == 0 else "1")
    print(f"//   " + ", ".join(arr))
    print()

if __name__ == "__main__":
    tests = [
        (1, 0), (1, 1), (1, 2),
        (2, 0), (2, 1), (2, 2),
        (3, 0), (3, 1), (3, 2), (3, 3),
        (4, 0), (4, 1), (4, 2),
        (5, 1),
        (8, 1),
    ]

    if "--print-table" in sys.argv:
        for N, d in tests:
            print_table(N, d)
        sys.exit(0)

    all_ok = True
    for N, d in tests:
        ok = check(N, d)
        status = "OK" if ok else "FAIL"
        print(f"  triu(N={N}, diag={d}) ... {status}")
        if not ok:
            all_ok = False

    if all_ok:
        print("\nAll checks pass — dnn.c tensor_triu matches PyTorch triu.")
    else:
        print("\nMISMATCH detected!")
        sys.exit(1)
