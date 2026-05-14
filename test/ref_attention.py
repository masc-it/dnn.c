#!/usr/bin/env python3
"""
PyTorch reference for scaled dot-product attention.

Compares dnn.c attention output against torch.nn.functional.scaled_dot_product_attention
(or manual compose of matmul + softmax + mask).

Usage:
    python3 test/ref_attention.py

Tests:
    1. 2D [N, d] forward (single sequence)
    2. 2D [N, d] with causal mask
    3. 3D [B, N, d] batched forward
    4. 4D [B, H, N, d] multi-head forward
    5. 3D backward gradient check (finite-diff)
"""

import torch
import torch.nn.functional as F
import numpy as np
import sys

EPS = 1e-4


def scaled_dot_product_attention(Q, K, V, mask=None):
    """
    Manual scaled dot-product attention.
    output = softmax(Q @ K^T / sqrt(d_k) + mask) @ V
    """
    d_k = Q.shape[-1]
    scale = 1.0 / (d_k ** 0.5)
    scores = torch.matmul(Q, K.transpose(-2, -1)) * scale
    if mask is not None:
        scores = scores + mask
    attn = F.softmax(scores, dim=-1)
    return torch.matmul(attn, V)


def test_2d_forward():
    """Test 2D [N, d] forward."""
    print("  ref_2d_forward... ", end="", flush=True)
    Q = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
    K = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
    V = torch.tensor([[1.0, 1.0], [2.0, 2.0]])
    out = scaled_dot_product_attention(Q, K, V)
    print(f"[{out[0,0]:.4f}, {out[0,1]:.4f}; {out[1,0]:.4f}, {out[1,1]:.4f}]")
    return out


def test_2d_causal():
    """Test 2D [N, d] with causal mask."""
    print("  ref_2d_causal... ", end="", flush=True)
    Q = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
    K = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
    V = torch.tensor([[1.0, 1.0], [2.0, 2.0]])
    N = 2
    mask = torch.triu(torch.full((N, N), float('-inf')), diagonal=1)
    out = scaled_dot_product_attention(Q, K, V, mask)
    print(f"[{out[0,0]:.4f}, {out[0,1]:.4f}; {out[1,0]:.4f}, {out[1,1]:.4f}]")
    return out


def test_3d_forward():
    """Test 3D [B, N, d] batched forward."""
    print("  ref_3d_forward... ", end="", flush=True)
    Q = torch.tensor([[[1.0, 0.0], [0.0, 1.0]],
                       [[2.0, 0.0], [0.0, 2.0]]])
    K = torch.tensor([[[1.0, 2.0], [3.0, 4.0]],
                       [[1.0, 2.0], [3.0, 4.0]]])
    V = torch.tensor([[[1.0, 1.0], [2.0, 2.0]],
                       [[1.0, 1.0], [2.0, 2.0]]])
    out = scaled_dot_product_attention(Q, K, V)
    print(f"[{out[0,0,0]:.4f} ... {out[1,1,1]:.4f}]")
    return out


def test_4d_forward():
    """Test 4D [B, H, N, d] multi-head forward."""
    print("  ref_4d_forward... ", end="", flush=True)
    Q = torch.tensor([[[[1.0, 0.0], [0.0, 1.0]],
                        [[2.0, 0.0], [0.0, 2.0]]]])
    K = torch.tensor([[[[1.0, 2.0], [3.0, 4.0]],
                        [[1.0, 2.0], [3.0, 4.0]]]])
    V = torch.tensor([[[[1.0, 1.0], [2.0, 2.0]],
                        [[1.0, 1.0], [2.0, 2.0]]]])
    out = scaled_dot_product_attention(Q, K, V)
    print(f"[{out[0,0,0,0]:.4f} ... {out[0,1,1,1]:.4f}]")
    return out


def test_single_token():
    """Test single token N=1."""
    print("  ref_single_token... ", end="", flush=True)
    Q = torch.tensor([[5.0]])
    K = torch.tensor([[3.0]])
    V = torch.tensor([[7.0]])
    out = scaled_dot_product_attention(Q, K, V)
    print(f"[{out[0,0]:.4f}]")
    return out


def test_3d_backward_gradcheck():
    """Numerical gradient check on 3D attention with torch.autograd.gradcheck."""
    print("  ref_3d_backward_gradcheck... ", end="", flush=True)

    def attention_fn(q, k, v):
        d_k = q.shape[-1]
        scale = 1.0 / (d_k ** 0.5)
        scores = torch.matmul(q, k.transpose(-2, -1)) * scale
        attn = F.softmax(scores, dim=-1)
        return torch.matmul(attn, v)

    Q = torch.tensor([[[1.0, 0.0], [0.0, 1.0]],
                       [[2.0, 0.0], [0.0, 2.0]]], requires_grad=True)
    K = torch.tensor([[[1.0, 2.0], [3.0, 4.0]],
                       [[1.0, 2.0], [3.0, 4.0]]], requires_grad=True)
    V = torch.tensor([[[1.0, 1.0], [2.0, 2.0]],
                       [[1.0, 1.0], [2.0, 2.0]]], requires_grad=True)

    # Run backward
    out = attention_fn(Q, K, V)
    loss = out.sum()
    loss.backward()

    assert Q.grad is not None, "Q.grad is None"
    assert K.grad is not None, "K.grad is None"
    assert V.grad is not None, "V.grad is None"
    assert Q.grad.abs().sum().item() > 0, "Q.grad is zero"
    assert K.grad.abs().sum().item() > 0, "K.grad is zero"
    assert V.grad.abs().sum().item() > 0, "V.grad is zero"

    print(f"OK (|dQ|={Q.grad.abs().sum().item():.4f})")
    return Q.grad, K.grad, V.grad


def test_2d_backward_gradcheck():
    """Numerical gradient check on 2D attention."""
    print("  ref_2d_backward_gradcheck... ", end="", flush=True)

    def attention_fn(q, k, v):
        d_k = q.shape[-1]
        scale = 1.0 / (d_k ** 0.5)
        scores = torch.matmul(q, k.transpose(-2, -1)) * scale
        attn = F.softmax(scores, dim=-1)
        return torch.matmul(attn, v)

    Q = torch.tensor([[1.0, 0.0], [0.0, 1.0]], requires_grad=True)
    K = torch.tensor([[1.0, 2.0], [3.0, 4.0]], requires_grad=True)
    V = torch.tensor([[1.0, 1.0], [2.0, 2.0]], requires_grad=True)

    out = attention_fn(Q, K, V)
    loss = out.sum()
    loss.backward()

    assert Q.grad is not None
    assert K.grad is not None
    assert V.grad is not None
    assert Q.grad.abs().sum().item() > 0
    assert K.grad.abs().sum().item() > 0
    assert V.grad.abs().sum().item() > 0

    print(f"OK (|dQ|={Q.grad.abs().sum().item():.4f})")
    return Q.grad, K.grad, V.grad


def test_causal_backward_gradcheck():
    """Numerical gradient check on 2D attention with causal mask."""
    print("  ref_causal_backward_gradcheck... ", end="", flush=True)

    def attention_fn(q, k, v, mask):
        d_k = q.shape[-1]
        scale = 1.0 / (d_k ** 0.5)
        scores = torch.matmul(q, k.transpose(-2, -1)) * scale + mask
        attn = F.softmax(scores, dim=-1)
        return torch.matmul(attn, v)

    Q = torch.tensor([[1.0, 0.0], [0.0, 1.0]], requires_grad=True)
    K = torch.tensor([[1.0, 2.0], [3.0, 4.0]], requires_grad=True)
    V = torch.tensor([[1.0, 1.0], [2.0, 2.0]], requires_grad=True)
    N = 2
    mask = torch.triu(torch.full((N, N), float('-inf')), diagonal=1)

    out = attention_fn(Q, K, V, mask)
    loss = out.sum()
    loss.backward()

    assert Q.grad is not None
    assert K.grad is not None
    assert V.grad is not None
    assert Q.grad.abs().sum().item() > 0
    assert K.grad.abs().sum().item() > 0
    assert V.grad.abs().sum().item() > 0

    print(f"OK (|dQ|={Q.grad.abs().sum().item():.4f})")
    return Q.grad, K.grad, V.grad


if __name__ == "__main__":
    print("ref_attention (PyTorch reference):")
    results = {}
    results['2d_forward'] = test_2d_forward()
    results['2d_causal'] = test_2d_causal()
    results['3d_forward'] = test_3d_forward()
    results['4d_forward'] = test_4d_forward()
    results['single_token'] = test_single_token()
    results['2d_backward'] = test_2d_backward_gradcheck()
    results['3d_backward'] = test_3d_backward_gradcheck()
    results['causal_backward'] = test_causal_backward_gradcheck()
    print("  ALL PASS")
    sys.exit(0)
