#!/usr/bin/env python3
"""Reference: KV cache append + get matched against PyTorch slicing semantics.

The KV cache is simply a pre-allocated buffer where we write new K/V
tokens at position seq_len along the sequence dim (dim=2), then get a
slice of the valid portion [B, H, seq_len, d_k].

Usage: uv run --with torch test/ref_kv_cache.py

Tests:
  1. Create cache, verify zero-filled
  2. Append one token, verify data and seq_len
  3. Append multiple tokens (2 then 1), verify accumulated data
  4. Multi-batch, multi-head append
  5. Fill to max_seq capacity
  6. Get before any append (empty slice)
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


def test_create():
    """Create cache, verify shape and zero-fill."""
    print("test_create...")
    B, H, max_seq, d_k = 2, 4, 512, 64
    K_cache = torch.zeros(B, H, max_seq, d_k)
    V_cache = torch.zeros(B, H, max_seq, d_k)

    assert K_cache.shape == (B, H, max_seq, d_k)
    assert V_cache.shape == (B, H, max_seq, d_k)
    assert K_cache.abs().max().item() == 0.0
    assert V_cache.abs().max().item() == 0.0
    print("  OK\n")


def test_append_one_token():
    """Append one token, verify data."""
    print("test_append_one_token...")
    B, H, max_seq, d_k = 1, 2, 16, 4
    K_cache = torch.zeros(B, H, max_seq, d_k)
    V_cache = torch.zeros(B, H, max_seq, d_k)
    seq_len = 0

    # K_new: [1, 2, 1, 4]
    K_new = torch.tensor([[
        [[1, 2, 3, 4]],
        [[9, 10, 11, 12]],
    ]], dtype=torch.float32)
    V_new = torch.tensor([[
        [[5, 6, 7, 8]],
        [[13, 14, 15, 16]],
    ]], dtype=torch.float32)

    # Append: write at position seq_len
    K_cache[:, :, seq_len:seq_len + 1, :] = K_new
    V_cache[:, :, seq_len:seq_len + 1, :] = V_new
    seq_len += 1

    # Get visible slice
    K_view = K_cache[:, :, :seq_len, :]
    V_view = V_cache[:, :, :seq_len, :]

    assert K_view.shape == (1, 2, 1, 4)
    assert V_view.shape == (1, 2, 1, 4)
    check("K_view", K_view, K_new)
    check("V_view", V_view, V_new)
    print("  OK\n")


def test_append_multiple_tokens():
    """Append 2 tokens then 1 token, verify accumulated data."""
    print("test_append_multiple_tokens...")
    B, H, max_seq, d_k = 1, 1, 8, 3
    K_cache = torch.zeros(B, H, max_seq, d_k)
    V_cache = torch.zeros(B, H, max_seq, d_k)
    seq_len = 0

    # Append 2 tokens
    K1 = torch.tensor([[[[1, 1, 1], [2, 2, 2]]]], dtype=torch.float32)  # [1,1,2,3]
    V1 = torch.tensor([[[[10, 10, 10], [20, 20, 20]]]], dtype=torch.float32)
    n1 = K1.shape[2]
    K_cache[:, :, seq_len:seq_len + n1, :] = K1
    V_cache[:, :, seq_len:seq_len + n1, :] = V1
    seq_len += n1
    assert seq_len == 2

    # Append 1 token
    K2 = torch.tensor([[[[3, 3, 3]]]], dtype=torch.float32)
    V2 = torch.tensor([[[[30, 30, 30]]]], dtype=torch.float32)
    n2 = K2.shape[2]
    K_cache[:, :, seq_len:seq_len + n2, :] = K2
    V_cache[:, :, seq_len:seq_len + n2, :] = V2
    seq_len += n2
    assert seq_len == 3

    K_view = K_cache[:, :, :seq_len, :]
    V_view = V_cache[:, :, :seq_len, :]

    assert K_view.shape == (1, 1, 3, 3)
    assert V_view.shape == (1, 1, 3, 3)

    check("K multi", K_view, torch.tensor([[[[1,1,1], [2,2,2], [3,3,3]]]], dtype=torch.float32))
    check("V multi", V_view, torch.tensor([[[[10,10,10], [20,20,20], [30,30,30]]]], dtype=torch.float32))
    print("  OK\n")


def test_append_multibatch_multihead():
    """Append with multiple batches and heads."""
    print("test_append_multibatch_multihead...")
    B, H, max_seq, d_k = 2, 3, 10, 2
    K_cache = torch.zeros(B, H, max_seq, d_k)
    V_cache = torch.zeros(B, H, max_seq, d_k)
    seq_len = 0

    # K: [2, 3, 2, 2]
    K_new = torch.arange(1, 2 * 3 * 2 * 2 + 1, dtype=torch.float32).reshape(2, 3, 2, 2)
    V_new = K_new * 100

    n_new = K_new.shape[2]
    K_cache[:, :, seq_len:seq_len + n_new, :] = K_new
    V_cache[:, :, seq_len:seq_len + n_new, :] = V_new
    seq_len += n_new
    assert seq_len == 2

    K_view = K_cache[:, :, :seq_len, :]
    V_view = V_cache[:, :, :seq_len, :]

    assert K_view.shape == (2, 3, 2, 2)
    check("K mb", K_view, K_new)
    check("V mb", V_view, V_new)

    # Spot-check b=1, h=2, n=1
    expected_k = K_new[1, 2, 1, 0]
    expected_v = V_new[1, 2, 1, 0]
    assert K_view[1, 2, 1, 0] == expected_k
    assert V_view[1, 2, 1, 0] == expected_v
    print("  OK\n")


def test_fill_to_max():
    """Append one at a time until full."""
    print("test_fill_to_max...")
    B, H, max_seq, d_k = 1, 1, 4, 2
    K_cache = torch.zeros(B, H, max_seq, d_k)
    V_cache = torch.zeros(B, H, max_seq, d_k)
    seq_len = 0

    for t in range(1, max_seq + 1):
        K_new = torch.tensor([[[[t, t * 10]]]], dtype=torch.float32)
        V_new = torch.tensor([[[[t * 100, t * 1000]]]], dtype=torch.float32)
        n = 1
        K_cache[:, :, seq_len:seq_len + n, :] = K_new
        V_cache[:, :, seq_len:seq_len + n, :] = V_new
        seq_len += n

    assert seq_len == max_seq

    K_view = K_cache[:, :, :seq_len, :]
    V_view = V_cache[:, :, :seq_len, :]

    exp_K = torch.tensor([[[[1,10], [2,20], [3,30], [4,40]]]], dtype=torch.float32)
    exp_V = torch.tensor([[[[100,1000], [200,2000], [300,3000], [400,4000]]]], dtype=torch.float32)
    check("K fill", K_view, exp_K)
    check("V fill", V_view, exp_V)
    print("  OK\n")


def test_empty_slice():
    """Get empty slice (seq_len=0) — shape [B, H, 0, d_k]."""
    print("test_empty_slice...")
    B, H, max_seq, d_k = 1, 1, 4, 2
    K_cache = torch.zeros(B, H, max_seq, d_k)
    V_cache = torch.zeros(B, H, max_seq, d_k)

    seq_len = 0
    K_view = K_cache[:, :, :seq_len, :]
    V_view = V_cache[:, :, :seq_len, :]

    assert K_view.shape == (1, 1, 0, 2)
    assert V_view.shape == (1, 1, 0, 2)
    assert K_view.numel() == 0
    assert V_view.numel() == 0
    print("  OK\n")


if __name__ == "__main__":
    test_create()
    test_append_one_token()
    test_append_multiple_tokens()
    test_append_multibatch_multihead()
    test_fill_to_max()
    test_empty_slice()
    print("ALL PASS")
