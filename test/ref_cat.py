#!/usr/bin/env python3
"""Reference: tensor_cat forward + backward matched against PyTorch.

Usage: uv run --with torch test/ref_cat.py

Tests:
  1. 1D cat forward/backward
  2. 2D cat along dim=0
  3. 2D cat along dim=1
  4. 3D cat along dim=1
  5. Cat with only a requiring grad
  6. Cat with same tensor (a == a)
  7. Cat chain with matmul
"""

import torch
import numpy as np

torch.manual_seed(42)
EPS = 1e-5


def check(name, got, expected):
    d = (got - expected).abs().max().item()
    ok = d < EPS
    print(f"  {name}: max_diff={d:.6f} {'OK' if ok else 'FAIL'}")
    assert ok, f"{name}: max_diff {d} >= {EPS}"


def test_cat_1d():
    """Simple 1D cat forward + backward."""
    print("test_cat_1d...")
    a = torch.tensor([1.0, 2.0, 3.0], requires_grad=True)
    b = torch.tensor([4.0, 5.0], requires_grad=True)

    out = torch.cat([a, b], dim=0)
    expected_fwd = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0])
    check("1D fwd", out, expected_fwd)

    loss = out.sum()
    loss.backward()

    check("1D a.grad", a.grad, torch.ones(3))
    check("1D b.grad", b.grad, torch.ones(2))
    print("  OK\n")


def test_cat_2d_dim0():
    """2D cat along dim=0 (vertical stacking)."""
    print("test_cat_2d_dim0...")
    a = torch.randn(2, 3, requires_grad=True)
    b = torch.randn(4, 3, requires_grad=True)

    out = torch.cat([a, b], dim=0)
    assert out.shape == (6, 3), f"shape mismatch: {out.shape}"

    loss = out.sum()
    loss.backward()

    check("2D dim0 a.grad", a.grad, torch.ones(2, 3))
    check("2D dim0 b.grad", b.grad, torch.ones(4, 3))
    print("  OK\n")


def test_cat_2d_dim1():
    """2D cat along dim=1 (horizontal stacking)."""
    print("test_cat_2d_dim1...")
    a = torch.randn(3, 2, requires_grad=True)
    b = torch.randn(3, 5, requires_grad=True)

    out = torch.cat([a, b], dim=1)
    assert out.shape == (3, 7), f"shape mismatch: {out.shape}"

    loss = out.sum()
    loss.backward()

    check("2D dim1 a.grad", a.grad, torch.ones(3, 2))
    check("2D dim1 b.grad", b.grad, torch.ones(3, 5))
    print("  OK\n")


def test_cat_3d_dim1():
    """3D cat along dim=1."""
    print("test_cat_3d_dim1...")
    a = torch.randn(2, 3, 4, requires_grad=True)
    b = torch.randn(2, 7, 4, requires_grad=True)

    out = torch.cat([a, b], dim=1)
    assert out.shape == (2, 10, 4), f"shape mismatch: {out.shape}"

    loss = out.sum()
    loss.backward()

    check("3D dim1 a.grad", a.grad, torch.ones(2, 3, 4))
    check("3D dim1 b.grad", b.grad, torch.ones(2, 7, 4))
    print("  OK\n")


def test_cat_only_a_grad():
    """Cat where only a requires grad."""
    print("test_cat_only_a_grad...")
    a = torch.randn(2, 3, requires_grad=True)
    b = torch.randn(4, 3, requires_grad=False)

    out = torch.cat([a, b], dim=0)
    loss = out.sum()
    loss.backward()

    check("only-a a.grad", a.grad, torch.ones(2, 3))
    assert b.grad is None, "b should not have grad"
    print("  OK\n")


def test_cat_self():
    """Cat with same tensor twice."""
    print("test_cat_self...")
    a = torch.tensor([1.0, 2.0], requires_grad=True)

    out = torch.cat([a, a], dim=0)
    assert out.shape == (4,), f"shape: {out.shape}"
    expected = torch.tensor([1.0, 2.0, 1.0, 2.0])
    check("self fwd", out, expected)

    loss = out.sum()
    loss.backward()

    # Each element of a contributes to 2 positions in output
    check("self a.grad", a.grad, torch.tensor([2.0, 2.0]))
    print("  OK\n")


def test_cat_chain_matmul():
    """Chain: cat -> matmul -> loss, grads flow through both paths."""
    print("test_cat_chain_matmul...")
    B, N, D = 2, 3, 4
    a = torch.randn(B, N, D, requires_grad=True)
    b = torch.randn(B, N, D, requires_grad=True)
    w = torch.randn(D, D, requires_grad=True)

    cat_out = torch.cat([a, b], dim=1)  # [B, 2N, D]
    out = cat_out @ w                     # [B, 2N, D]
    loss = out.sum()
    loss.backward()

    # Verify shapes and grads exist
    assert a.grad is not None
    assert b.grad is not None
    assert w.grad is not None
    assert a.grad.shape == (B, N, D)
    assert b.grad.shape == (B, N, D)

    # Sum of a.grad == sum of ones @ w.T over the a slice
    cat_grad = torch.ones(B, 2*N, D) @ w.T
    expected_a_grad = cat_grad[:, :N, :]
    expected_b_grad = cat_grad[:, N:, :]
    check("chain a.grad", a.grad, expected_a_grad)
    check("chain b.grad", b.grad, expected_b_grad)
    print("  OK\n")


def test_cat_forward_values():
    """2D cat dim=1 with fixed values — exact match."""
    print("test_cat_forward_values...")
    a = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
    b = torch.tensor([[5.0, 6.0, 7.0], [8.0, 9.0, 10.0]])

    out = torch.cat([a, b], dim=1)
    expected = torch.tensor([
        [1.0, 2.0, 5.0, 6.0, 7.0],
        [3.0, 4.0, 8.0, 9.0, 10.0],
    ])
    check("fwd exact", out, expected)
    print("  OK\n")


if __name__ == "__main__":
    test_cat_1d()
    test_cat_2d_dim0()
    test_cat_2d_dim1()
    test_cat_3d_dim1()
    test_cat_only_a_grad()
    test_cat_self()
    test_cat_chain_matmul()
    test_cat_forward_values()
    print("ALL PASS")
