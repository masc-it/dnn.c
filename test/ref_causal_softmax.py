#!/usr/bin/env python3
"""Reference: fused causal softmax for dnn.c.

Pure Python (stdlib only). Implements forward & backward of
causal softmax. Prints reference values for C test assertions.
"""

import math
import random
import sys

EPS = 1e-5


def softmax_1d(x):
    mx = max(x)
    ex = [math.exp(v - mx) for v in x]
    s = sum(ex)
    return [v / s for v in ex]


def causal_softmax_2d(scores):
    N = len(scores)
    out = [[0.0] * N for _ in range(N)]
    for i in range(N):
        row = scores[i][:i + 1]
        sm = softmax_1d(row)
        for j, val in enumerate(sm):
            out[i][j] = val
    return out


def causal_softmax_backward_2d(scores, grad):
    N = len(scores)
    sm = causal_softmax_2d(scores)
    d = [[0.0] * N for _ in range(N)]
    for i in range(N):
        dot = sum(sm[i][j] * grad[i][j] for j in range(i + 1))
        for j in range(i + 1):
            d[i][j] = sm[i][j] * (grad[i][j] - dot)
    return d, sm  # return sm too for convenience


def causal_softmax_ndim(scores):
    """Recursive for nested list with ndim >= 2."""
    if isinstance(scores[0][0], list):
        return [causal_softmax_ndim(sub) for sub in scores]
    return causal_softmax_2d(scores)


def flatten_2d(m):
    return [v for row in m for v in row]


if __name__ == "__main__":
    random.seed(42)
    N = 4

    # --- Generate scores ---
    scores = [[random.gauss(0, 1) for _ in range(N)] for _ in range(N)]

    # --- Forward ---
    out = causal_softmax_2d(scores)

    # Verify forward properties
    for i in range(N):
        assert abs(sum(out[i][:i + 1]) - 1.0) < EPS, f"row {i} softmax sum != 1"
        for j in range(i + 1, N):
            assert abs(out[i][j]) < EPS, f"out[{i}][{j}] != 0"

    # --- Backward with non-uniform gradient ---
    # Use gradient = [1, 2, 3, 4] per element along cols (distinct per row)
    grad_in = [[float(j + 1) for j in range(N)] for _ in range(N)]
    grad_out, sm = causal_softmax_backward_2d(scores, grad_in)

    # Quick sanity: for each row, sum(gradient * sm) = dot, and
    # dL/dx values should not be all zero (unlike unit gradient case)
    for i in range(N):
        dot = sum(sm[i][j] * grad_in[i][j] for j in range(i + 1))
        for j in range(i + 1):
            expected = sm[i][j] * (grad_in[i][j] - dot)
            assert abs(grad_out[i][j] - expected) < EPS
        for j in range(i + 1, N):
            assert abs(grad_out[i][j]) < EPS

    # --- 4D test ---
    B, H = 2, 3
    scores_4d = [[[[random.gauss(0, 1) for _ in range(N)]
                   for _ in range(N)] for _ in range(H)] for _ in range(B)]
    out_4d = causal_softmax_ndim(scores_4d)
    for b in range(B):
        for h in range(H):
            for i in range(N):
                s = sum(out_4d[b][h][i][:i + 1])
                assert abs(s - 1.0) < EPS, f"4d[{b}][{h}][{i}] sum={s}"

    # --- Print reference for C test ---
    # We'll test forward match + backward match vs gradient with col values
    print(f"int N = {N};")
    print()
    print("/* scores (row-major) */")
    flat_scores = flatten_2d(scores)
    print("float ref_scores[] = {" +
          ", ".join(f"{v:.10f}f" for v in flat_scores) + "};")
    print()
    print("/* expected forward output */")
    flat_out = flatten_2d(out)
    print("float ref_out[] = {" +
          ", ".join(f"{v:.10f}f" for v in flat_out) + "};")
    print()
    print("/* upstream gradient for backward test (j+1 per col) */")
    flat_grad_in = flatten_2d(grad_in)
    print("float ref_grad_in[] = {" +
          ", ".join(f"{v:.1f}f" for v in flat_grad_in) + "};")
    print()
    print("/* expected backward gradient w.r.t scores */")
    flat_grad_out = flatten_2d(grad_out)
    print("float ref_grad_out[] = {" +
          ", ".join(f"{v:.10f}f" for v in flat_grad_out) + "};")

    print()
    print("/* sanity: verify grad_out rows have zero-sum over visible */")
    for i in range(N):
        s = sum(grad_out[i][:i + 1])
        print(f"/*   row {i} visible sum = {s:.6e} */")
    for i in range(N):
        for j in range(i + 1, N):
            assert abs(grad_out[i][j]) < EPS, f"grad_out[{i}][{j}] != 0"

    print()
    print("All checks pass.", file=sys.stderr)
