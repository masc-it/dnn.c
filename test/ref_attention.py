#!/usr/bin/env python3
"""
PyTorch reference for fused causal attention.

Computes:
    O = causal_softmax( (Q @ K^T) / sqrt(d_head) ) @ V

Verifies gradients against torch.autograd.
Used to generate expected values for the C test suite.
"""
import torch
import torch.nn.functional as F
import numpy as np
import sys

def causal_softmax(x):
    """Causal softmax over last dim: rows i attend only to j <= i."""
    N = x.shape[-1]
    mask = torch.triu(torch.full((N, N), float('-inf'), device=x.device), diagonal=1)
    return F.softmax(x + mask, dim=-1)

def attention_ref(q, k, v):
    """
    Fused causal attention: O = causal_softmax(Q@K^T/√d) @ V.

    q, k, v: shape [B, H, N, d_head]
    Returns:  O shape [B, H, N, d_head]
    """
    d_head = q.shape[-1]
    scale = d_head ** -0.5
    scores = torch.matmul(q, k.transpose(-2, -1)) * scale
    attn = causal_softmax(scores)
    return torch.matmul(attn, v), attn

def test_shapes():
    """Verify forward output for small deterministic case."""
    torch.manual_seed(42)
    B, H, N, d = 2, 3, 4, 8
    q = torch.randn(B, H, N, d, requires_grad=True)
    k = torch.randn(B, H, N, d, requires_grad=True)
    v = torch.randn(B, H, N, d, requires_grad=True)

    out, attn = attention_ref(q, k, v)

    # Check shapes
    assert out.shape == (B, H, N, d), f"out shape mismatch: {out.shape}"
    assert attn.shape == (B, H, N, N), f"attn shape mismatch: {attn.shape}"

    # Check causal mask: each row i should be zero for j > i
    for b in range(B):
        for h in range(H):
            for i in range(N):
                for j in range(i+1, N):
                    assert attn[b,h,i,j].item() == 0.0, \
                        f"causal mask violation at ({b},{h},{i},{j})"

    # Check each row sums to 1
    for b in range(B):
        for h in range(H):
            for i in range(N):
                row_sum = attn[b,h,i,:i+1].sum().item()
                assert abs(row_sum - 1.0) < 1e-5, \
                    f"row sum != 1 at ({b},{h},{i}): {row_sum}"

    print(f"  shapes test: OK  (B={B}, H={H}, N={N}, d={d})")
    return out, attn, (q, k, v)

def test_gradients():
    """Verify gradients via autograd against manual formulas."""
    torch.manual_seed(123)
    B, H, N, d = 1, 1, 3, 4
    q = torch.randn(B, H, N, d, requires_grad=True)
    k = torch.randn(B, H, N, d, requires_grad=True)
    v = torch.randn(B, H, N, d, requires_grad=True)

    out, attn = attention_ref(q, k, v)
    loss = out.sum()
    loss.backward()

    # Manual gradient computation for verification
    dO = torch.ones_like(out)  # grad of sum
    d_head = d

    dV_manual = torch.matmul(attn.transpose(-2, -1), dO)

    dP_manual = torch.matmul(dO, v.transpose(-2, -1))

    # causal softmax backward
    dS_manual = torch.zeros_like(dP_manual)
    for b in range(B):
        for h in range(H):
            for i in range(N):
                dot = (attn[b,h,i,:i+1] * dP_manual[b,h,i,:i+1]).sum()
                dS_manual[b,h,i,:i+1] = attn[b,h,i,:i+1] * (dP_manual[b,h,i,:i+1] - dot)
                # j > i stays 0

    scale = d_head ** -0.5
    dS_manual = dS_manual * scale

    dQ_manual = torch.matmul(dS_manual, k)
    dK_manual = torch.matmul(dS_manual.transpose(-2, -1), q)

    tol = 1e-5
    for name, got, expected in [
        ("dQ", q.grad, dQ_manual),
        ("dK", k.grad, dK_manual),
        ("dV", v.grad, dV_manual),
    ]:
        diff = (got - expected).abs().max().item()
        status = "OK" if diff < tol else "FAIL"
        print(f"  {name} max diff: {diff:.2e}  {status}")
        assert diff < tol, f"{name} gradient mismatch: {diff:.2e}"

    print("  gradients test: OK")

def test_multiple_heads():
    """Verify with multiple heads and larger dimensions."""
    torch.manual_seed(456)
    B, H, N, d = 2, 4, 8, 16
    q = torch.randn(B, H, N, d, requires_grad=True)
    k = torch.randn(B, H, N, d, requires_grad=True)
    v = torch.randn(B, H, N, d, requires_grad=True)

    out, attn = attention_ref(q, k, v)
    loss = out.sum()
    loss.backward()

    assert q.grad is not None
    assert k.grad is not None
    assert v.grad is not None
    assert q.grad.shape == (B, H, N, d)
    assert k.grad.shape == (B, H, N, d)
    assert v.grad.shape == (B, H, N, d)

    # Verify grad components are non-zero and finite
    for name, g in [("Q", q.grad), ("K", k.grad), ("V", v.grad)]:
        assert torch.isfinite(g).all(), f"{name} grad has inf/nan"
        assert g.abs().sum().item() > 0, f"{name} grad is all zero"

    print(f"  multi-head test: OK  (B={B}, H={H}, N={N}, d={d})")

def test_reference_values():
    """Generate reference values for C test, printed as C arrays."""
    torch.manual_seed(789)
    B, H, N, d = 1, 1, 3, 4

    q = torch.randn(B, H, N, d, requires_grad=True)
    k = torch.randn(B, H, N, d, requires_grad=True)
    v = torch.randn(B, H, N, d, requires_grad=True)

    out, attn = attention_ref(q, k, v)
    loss = out.sum()
    loss.backward()

    print("\n  Reference values (for test_attention.c):")
    print(f"  // seed=789, B={B}, H={H}, N={N}, d={d}")
    print(f"  // q_data = {_fmt_tensor(q.detach())}")
    print(f"  // k_data = {_fmt_tensor(k.detach())}")
    print(f"  // v_data = {_fmt_tensor(v.detach())}")
    print(f"  // out_data = {_fmt_tensor(out.detach())}")
    print(f"  // dq_data = {_fmt_tensor(q.grad)}")
    print(f"  // dk_data = {_fmt_tensor(k.grad)}")
    print(f"  // dv_data = {_fmt_tensor(v.grad)}")

def _fmt_tensor(t):
    """Format small tensor as C float array literal."""
    flat = t.detach().cpu().numpy().flatten()
    items = ", ".join(f"{v:.8f}f" for v in flat)
    return "{" + items + "}"

if __name__ == "__main__":
    print("PyTorch reference: fused causal attention")
    print("=" * 50)
    test_shapes()
    test_gradients()
    test_multiple_heads()
    test_reference_values()
    print("=" * 50)
    print("All tests passed.")
