#!/usr/bin/env python3
"""Reference: tensor_slice forward + backward matched against PyTorch.

Usage: uv run --with torch python test/ref_slice.py

Tests:
  1. Basic 2D slice forward/backward
  2. 3D slice along last dim (QKV split pattern)
  3. 3D slice dim=1 pattern
  4. Non-contiguous parent (after transpose)
  5. Chain: slice -> matmul with grad check
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


def test_slice_2d():
    """Slice a 2D [4,6] tensor along dim=0 and dim=1."""
    print("test_slice_2d...")
    x = torch.randn(4, 6, requires_grad=True)

    # slice along dim=0: rows 1..3
    s0 = x[1:3, :]
    loss0 = s0.sum()
    loss0.backward()
    g0 = x.grad.clone()

    x.grad = None
    # slice along dim=1: cols 2..5
    s1 = x[:, 2:5]
    loss1 = s1.sum()
    loss1.backward()
    g1 = x.grad.clone()

    # Verify: d_loss/d_x[i,j] = 1 if in slice, 0 otherwise
    expected_g0 = torch.zeros_like(x)
    expected_g0[1:3, :] = 1.0
    check("slice dim=0 grad", g0, expected_g0)

    expected_g1 = torch.zeros_like(x)
    expected_g1[:, 2:5] = 1.0
    check("slice dim=1 grad", g1, expected_g1)

    # Also check forward values
    check("slice dim=0 fwd", s0, x[1:3, :])
    check("slice dim=1 fwd", s1, x[:, 2:5])
    print("  OK\n")


def test_slice_3d_last_dim():
    """Slice 3D [B,N,3D] along last dim — QKV split pattern."""
    print("test_slice_3d_last_dim (QKV split)...")
    B, N, D = 2, 4, 8
    x = torch.randn(B, N, 3*D, requires_grad=True)

    q = x[:, :, 0:D]
    k = x[:, :, D:2*D]
    v = x[:, :, 2*D:3*D]

    # forward: verify values
    check("q slice", q, x[:, :, :D])
    check("k slice", k, x[:, :, D:2*D])
    check("v slice", v, x[:, :, 2*D:])

    # backward: sum loss through all three slices
    loss = q.sum() + k.sum() + v.sum()
    loss.backward()

    expected_grad = torch.ones_like(x)  # each element appears in exactly one slice
    check("qkv split grad", x.grad, expected_grad)
    print("  OK\n")


def test_slice_3d_dim1():
    """Slice 3D along dim=1."""
    print("test_slice_3d_dim1...")
    x = torch.randn(2, 6, 4, requires_grad=True)

    s = x[:, 2:5, :]
    loss = s.sum()
    loss.backward()

    expected = torch.zeros_like(x)
    expected[:, 2:5, :] = 1.0
    check("dim=1 grad", x.grad, expected)
    print("  OK\n")


def test_slice_noncontiguous():
    """Slice after transpose — non-contiguous parent."""
    print("test_slice_noncontiguous...")
    x = torch.randn(3, 5, requires_grad=True)
    xt = x.T  # [5, 3], non-contiguous

    s = xt[1:4, :]
    loss = s.sum()
    loss.backward()

    # d_loss/d_x[i,j] = 1 if x.T [1:4,:] contains this element
    # x.T[1:4, :] corresponds to x[:, 1:4]
    expected = torch.zeros_like(x)
    expected[:, 1:4] = 1.0
    check("noncontig slice grad", x.grad, expected)
    print("  OK\n")


def test_slice_chain():
    """Chain: slice -> matmul with grad through both."""
    print("test_slice_chain...")
    B, N, D = 2, 3, 5
    # Simulate QKV projection output
    x = torch.randn(B, N, 3*D, requires_grad=True)
    w = torch.randn(D, D, requires_grad=True)

    q = x[:, :, 0:D]
    # matmul: [B, N, D] @ [D, D] -> [B, N, D]
    out = q @ w
    loss = out.sum()
    loss.backward()

    # x.grad should be non-zero only in the q slice region
    expected_x_grad = torch.zeros_like(x)
    # gradient of out = q @ w w.r.t q is ones @ w.T
    q_grad = torch.ones(B, N, D) @ w.T
    expected_x_grad[:, :, 0:D] = q_grad
    check("chain x.grad", x.grad, expected_x_grad)

    # w.grad should be q_flat.T @ ones_flat
    q_val = x[:, :, 0:D].detach()
    expected_w_grad = q_val.reshape(-1, D).T @ torch.ones(B*N, D)
    check("chain w.grad", w.grad, expected_w_grad)
    print("  OK\n")


def test_slice_embedding_chain():
    """Chain: embedding -> slice -> loss, verify grads flow to table."""
    print("test_slice_embedding_chain...")
    V, D = 10, 4
    table = torch.randn(V, D, requires_grad=True)
    ids = torch.tensor([2, 5, 1, 7], dtype=torch.long)

    emb = table[ids]  # [4, D]
    # Slice along dim=0: first 2 rows
    s = emb[0:2, :]
    loss = s.sum()
    loss.backward()

    # Only ids[0]=2 and ids[1]=5 get gradient
    expected = torch.zeros_like(table)
    expected[2] = 1.0  # first row of emb -> table row 2
    expected[5] = 1.0  # second row of emb -> table row 5
    check("embed slice table grad", table.grad, expected)
    print("  OK\n")


if __name__ == "__main__":
    test_slice_2d()
    test_slice_3d_last_dim()
    test_slice_3d_dim1()
    test_slice_noncontiguous()
    test_slice_chain()
    test_slice_embedding_chain()
    print("ALL PASS")
