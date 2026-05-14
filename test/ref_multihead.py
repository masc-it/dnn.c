#!/usr/bin/env python3
"""
PyTorch reference for multi-head split/merge operations.

Compares dnn.c tensor_split_heads / tensor_merge_heads against
PyTorch view + transpose equivalent.

Tests:
    1. split_heads: [B,N,H*d_k] → [B,H,N,d_k]
    2. merge_heads: [B,H,N,d_k] → [B,N,H*d_k]
    3. split → merge roundtrip (identity)
    4. merge → split roundtrip (identity)
    5. gradient flow through split
    6. gradient flow through merge

Usage:
    python3 test/ref_multihead.py
"""

import torch
import numpy as np
import sys

EPS = 1e-4


def split_heads_ref(x, H):
    """PyTorch reference: [B, N, H*d_k] → [B, H, N, d_k]"""
    B, N, D = x.shape
    assert D % H == 0, "last dim must be divisible by H"
    d_k = D // H
    # [B, N, H, d_k]
    v = x.view(B, N, H, d_k)
    # [B, H, N, d_k]
    return v.transpose(1, 2).contiguous()


def merge_heads_ref(x):
    """PyTorch reference: [B, H, N, d_k] → [B, N, H*d_k]"""
    B, H, N, d_k = x.shape
    # [B, N, H, d_k]
    v = x.transpose(1, 2).contiguous()
    # [B, N, H*d_k]
    return v.reshape(B, N, H * d_k)


def test_split_basic():
    """[1, 2, 4] → [1, 2, 2, 2]"""
    print("  ref_split_basic... ", end="", flush=True)
    x = torch.tensor([[[1.0, 2.0, 3.0, 4.0],
                        [5.0, 6.0, 7.0, 8.0]]])
    out = split_heads_ref(x, 2)
    print(f"shape={tuple(out.shape)} [{out[0,0,0,0].item():.0f},{out[0,0,1,0].item():.0f}..{out[0,1,1,1].item():.0f}]")
    return out


def test_split_batched():
    """[2, 2, 4] → [2, 2, 2, 2]"""
    print("  ref_split_batched... ", end="", flush=True)
    x = torch.tensor(range(16), dtype=torch.float32).reshape(2, 2, 4)
    out = split_heads_ref(x, 2)
    print(f"shape={tuple(out.shape)} [{out[0,0,0,0].item():.0f}...{out[1,1,1,1].item():.0f}]")
    return out


def test_split_h1():
    """[1, 2, 2] → [1, 1, 2, 2] (H=1 degenerate)"""
    print("  ref_split_h1... ", end="", flush=True)
    x = torch.tensor([[[1.0, 2.0], [3.0, 4.0]]])
    out = split_heads_ref(x, 1)
    print(f"shape={tuple(out.shape)} [{out[0,0,0,0].item():.0f}..{out[0,0,1,1].item():.0f}]")
    return out


def test_merge_basic():
    """[1, 2, 2, 2] → [1, 2, 4]"""
    print("  ref_merge_basic... ", end="", flush=True)
    x = torch.tensor([[[[1.0, 2.0], [3.0, 4.0]],
                        [[5.0, 6.0], [7.0, 8.0]]]])
    out = merge_heads_ref(x)
    print(f"shape={tuple(out.shape)} [{out[0,0,0].item():.0f},{out[0,0,1].item():.0f}..{out[0,1,3].item():.0f}]")
    return out


def test_merge_batched():
    """[2, 2, 2, 2] → [2, 2, 4]"""
    print("  ref_merge_batched... ", end="", flush=True)
    x = torch.tensor(range(16), dtype=torch.float32).reshape(2, 2, 2, 2)
    out = merge_heads_ref(x)
    print(f"shape={tuple(out.shape)} [{out[0,0,0].item():.0f}...{out[1,1,3].item():.0f}]")
    return out


def test_split_merge_roundtrip():
    """split then merge = identity"""
    print("  ref_split_merge_roundtrip... ", end="", flush=True)
    x = torch.tensor(range(12), dtype=torch.float32).reshape(1, 3, 4)
    split = split_heads_ref(x, 2)
    merged = merge_heads_ref(split)
    diff = (x - merged).abs().max().item()
    assert diff < EPS, f"split→merge roundtrip error: {diff}"
    print(f"OK (max_diff={diff:.1e})")
    return merged


def test_merge_split_roundtrip():
    """merge then split = identity"""
    print("  ref_merge_split_roundtrip... ", end="", flush=True)
    x = torch.tensor(range(16), dtype=torch.float32).reshape(2, 2, 2, 2)
    merged = merge_heads_ref(x)
    split = split_heads_ref(merged, 2)
    diff = (x - split).abs().max().item()
    assert diff < EPS, f"merge→split roundtrip error: {diff}"
    print(f"OK (max_diff={diff:.1e})")
    return split


def test_split_grad():
    """gradient flows through split"""
    print("  ref_split_grad... ", end="", flush=True)
    x = torch.tensor([[[1.0, 2.0, 3.0, 4.0],
                        [5.0, 6.0, 7.0, 8.0]]], requires_grad=True)
    split = split_heads_ref(x, 2)
    loss = split.sum()
    loss.backward()
    assert x.grad is not None
    assert x.grad.abs().sum().item() > 0
    print(f"OK (|dX|={x.grad.abs().sum().item():.4f})")
    return x.grad


def test_merge_grad():
    """gradient flows through merge"""
    print("  ref_merge_grad... ", end="", flush=True)
    x = torch.tensor([[[[1.0, 2.0], [3.0, 4.0]],
                        [[5.0, 6.0], [7.0, 8.0]]]], requires_grad=True)
    merged = merge_heads_ref(x)
    loss = merged.sum()
    loss.backward()
    assert x.grad is not None
    assert x.grad.abs().sum().item() > 0
    print(f"OK (|dX|={x.grad.abs().sum().item():.4f})")
    return x.grad


def test_split_various_shapes():
    """[1, 2, 12] → [1, 4, 2, 3] (H=4, d_k=3)"""
    print("  ref_split_various_shapes... ", end="", flush=True)
    x = torch.tensor(range(24), dtype=torch.float32).reshape(1, 2, 12)
    out = split_heads_ref(x, 4)
    expected = torch.tensor([
        [[[0,1,2], [12,13,14]],
         [[3,4,5], [15,16,17]],
         [[6,7,8], [18,19,20]],
         [[9,10,11], [21,22,23]]]
    ])
    diff = (out - expected).abs().max().item()
    assert diff < EPS, f"split various shapes error: {diff}"
    print(f"OK (max_diff={diff:.1e})")
    return out


def test_merge_various_shapes():
    """[1, 4, 2, 3] → [1, 2, 12]"""
    print("  ref_merge_various_shapes... ", end="", flush=True)
    x = torch.tensor(range(24), dtype=torch.float32).reshape(1, 4, 2, 3)
    out = merge_heads_ref(x)
    expected = torch.tensor([
        [[0,1,2,6,7,8,12,13,14,18,19,20],
         [3,4,5,9,10,11,15,16,17,21,22,23]]
    ])
    diff = (out - expected).abs().max().item()
    assert diff < EPS, f"merge various shapes error: {diff}"
    print(f"OK (max_diff={diff:.1e})")
    return out


if __name__ == "__main__":
    print("ref_multihead (PyTorch reference):")
    results = {}
    results['split_basic'] = test_split_basic()
    results['split_batched'] = test_split_batched()
    results['split_h1'] = test_split_h1()
    results['merge_basic'] = test_merge_basic()
    results['merge_batched'] = test_merge_batched()
    results['split_merge_roundtrip'] = test_split_merge_roundtrip()
    results['merge_split_roundtrip'] = test_merge_split_roundtrip()
    results['split_grad'] = test_split_grad()
    results['merge_grad'] = test_merge_grad()
    results['split_various_shapes'] = test_split_various_shapes()
    results['merge_various_shapes'] = test_merge_various_shapes()
    print("  ALL PASS")
    sys.exit(0)
