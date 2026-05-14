#!/usr/bin/env python3
"""PyTorch reference for fused tensor_silu (SiLU/Swish).

Verifies forward values and backward gradients match the C tests in
test_autograd.c.  Run: python3 test/ref_silu.py

Expected output (numerical values used in test_silu_scalar / test_silu_array):
"""

import torch
import math

EPS = 1e-5

def silu_ref(x):
    """SiLU/Swish: x * sigmoid(x)"""
    return x * torch.sigmoid(x)


def check(label, got, expected):
    if abs(got - expected) > EPS:
        print(f"  FAIL: {label}: got {got:.8f}, expected {expected:.8f}")
    else:
        print(f"  OK:   {label}: {got:.8f}")


def test_silu_scalar():
    print("\n=== test_silu_scalar ===")
    xs = [0.0, 1.0, -1.0, 2.0, -2.0, 3.0, -3.0, 0.5, -0.5]

    x = torch.tensor(xs, requires_grad=True)
    y = silu_ref(x)
    loss = y.sum()
    loss.backward()

    print("Forward (SiLU values):")
    for i, val in enumerate(xs):
        check(f"silu({val})", y[i].item(),
              x[i].item() / (1.0 + math.exp(-x[i].item())))

    print("Gradients:")
    expected_fwd = [0.00000000, 0.73105860, -0.26894143, 1.76159406, -0.23840584,
                    2.85772252, -0.14227761, 0.31122968, -0.18877034]
    expected_grad = [0.50000000, 0.92767054, 0.07232948, 1.09078431, -0.09078425,
                     1.08810413, -0.08810411, 0.73996121, 0.26003882]
    for i, val in enumerate(xs):
        check(f"silu'({val})", x.grad[i].item(), expected_grad[i])


def test_silu_array():
    print("\n=== test_silu_array ===")
    vals = [-3.0, -1.0, 0.0, 2.0, 4.0]

    x = torch.tensor(vals, requires_grad=True)
    y = silu_ref(x)
    loss = y.sum()
    loss.backward()

    expected_fwd = [-0.14227761, -0.26894143, 0.00000000, 1.76159406, 3.92805505]
    expected_grad = [-0.08810411, 0.07232948, 0.50000000, 1.09078431, 1.05266464]

    print("Forward:")
    for i, val in enumerate(vals):
        check(f"silu({val})", y[i].item(), expected_fwd[i])

    print("Gradients:")
    for i, val in enumerate(vals):
        check(f"silu'({val})", x.grad[i].item(), expected_grad[i])


def test_silu_chain():
    print("\n=== test_silu_chain ===")
    # c = silu(a) + a,  dc/da = silu'(a) + 1
    a = torch.tensor([1.0], requires_grad=True)
    c = silu_ref(a) + a
    c.backward()

    silu_grad = 0.92767054  # silu'(1)
    expected = silu_grad + 1.0
    check("silu'(1) + 1", a.grad.item(), expected)


def test_silu_no_grad():
    print("\n=== test_silu_no_grad ===")
    with torch.no_grad():
        x = torch.tensor([3.0])
        y = silu_ref(x)
        ref = 3.0 / (1.0 + math.exp(-3.0))
        check("silu(3) no-grad", y.item(), ref)


if __name__ == "__main__":
    print("PyTorch SiLU (Swish) reference")
    print(f"PyTorch version: {torch.__version__}")
    test_silu_scalar()
    test_silu_array()
    test_silu_chain()
    test_silu_no_grad()
    print("\nAll checks done.\n")
