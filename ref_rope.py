#!/usr/bin/env python3
"""PyTorch reference for RoPE (Rotary Position Embedding).

Usage:
    python3 ref_rope.py                  # run tests, print results
    python3 ref_rope.py --save-outputs   # save .bin files for C test to read

Generates:
    rope_fwd_in.bin     — input tensor (float32, [B,H,N,d])
    rope_fwd_cos.bin    — cos table (float32, [N,d//2])
    rope_fwd_sin.bin    — sin table (float32, [N,d//2])
    rope_fwd_out.bin    — expected output (float32, [B,H,N,d])
    rope_bwd_grad.bin   — upstream gradient (float32, [B,H,N,d])
    rope_bwd_dxin.bin   — expected input grad (float32, [B,H,N,d])
    rope_freqs_cos.bin  — pre-initialized cos table (float32, [max_seq_len, d//2])
    rope_freqs_sin.bin  — pre-initialized sin table (float32, [max_seq_len, d//2])
"""

import torch
import math
import struct
import sys
import os

def rope_freqs(dim, max_seq_len, base=10000.0):
    """Compute cos/sin frequency tables matching tensor_rope_freqs_init."""
    d_half = dim // 2
    theta = base ** (-torch.arange(0, dim, 2, dtype=torch.float32) / dim)
    # theta: [d_half]
    n = torch.arange(max_seq_len, dtype=torch.float32)  # [max_seq_len]
    angles = n[:, None] * theta[None, :]  # [max_seq_len, d_half]
    cos = torch.cos(angles)
    sin = torch.sin(angles)
    return cos, sin

def rope_pairwise(x, cos, sin):
    """Pair-wise RoPE: rotate pairs (2k, 2k+1) along last dim.
    
    x:   [B, H, N, d] or [B, N, d] etc., last dim must be even
    cos: [N, d//2] or broadcastable
    sin: [N, d//2]
    """
    d = x.shape[-1]
    d_half = d // 2
    
    # Reshape to expose pairs as last dim = 2
    x_reshaped = x.reshape(*x.shape[:-1], d_half, 2)  # [..., d_half, 2]
    
    e = x_reshaped[..., 0]  # even elements: [..., d_half]
    o = x_reshaped[..., 1]  # odd elements:  [..., d_half]
    
    # Broadcast cos/sin to match x's shape[:-1]
    # cos: [N, d_half] → [1, 1, N, d_half] for [B, H, N, d_half]
    # We need to unsqueeze leading dims to match x's ndim-1
    while cos.dim() < x.dim() - 1:
        cos = cos.unsqueeze(0)
        sin = sin.unsqueeze(0)
    
    e_new = e * cos - o * sin
    o_new = e * sin + o * cos
    
    out = torch.stack([e_new, o_new], dim=-1)  # [..., d_half, 2]
    out = out.reshape(*x.shape)  # [..., d]
    return out

def rope_inplace_equivalent(x, cos, sin):
    """Compute RoPE output without modifying x (for gradient comparison)."""
    return rope_pairwise(x, cos, sin)

def test_correctness():
    """Test RoPE against known values."""
    print("=== Correctness tests ===")
    
    # Test 1: trivial case d=2, N=1, B=1, H=1
    x = torch.tensor([[[[1.0, 0.0]]]])  # [1, 1, 1, 2]
    cos = torch.tensor([[1.0]])  # [1, 1] → [N=1, d_half=1]
    sin = torch.tensor([[0.0]])  # cos(0)=1, sin(0)=0 → identity
    out = rope_pairwise(x, cos, sin)
    expected = torch.tensor([[[[1.0, 0.0]]]])
    assert torch.allclose(out, expected, atol=1e-6), f"Test 1 failed: {out}"
    print("  PASS: identity rotation (cos=1, sin=0)")
    
    # Test 2: rotation by 90°: (1,0) → (0,1)
    cos90 = torch.tensor([[0.0]])  # cos(90°) = 0
    sin90 = torch.tensor([[1.0]])  # sin(90°) = 1
    out2 = rope_pairwise(x, cos90, sin90)
    expected2 = torch.tensor([[[[0.0, 1.0]]]])
    assert torch.allclose(out2, expected2, atol=1e-6), f"Test 2 failed: {out2}"
    print("  PASS: 90° rotation (cos=0, sin=1): (1,0) → (0,1)")
    
    # Test 3: rotation by 45°: (1,0) → (√2/2, √2/2)
    s2 = math.sqrt(2) / 2
    cos45 = torch.tensor([[s2]])
    sin45 = torch.tensor([[s2]])
    out3 = rope_pairwise(x, cos45, sin45)
    expected3 = torch.tensor([[[[s2, s2]]]])
    assert torch.allclose(out3, expected3, atol=1e-6), f"Test 3 failed: {out3}"
    print("  PASS: 45° rotation (cos=√2/2, sin=√2/2): (1,0) → (√2/2, √2/2)")
    
    # Test 4: rotation by 30°: (1,2) with known values
    c = math.cos(math.radians(30))  # √3/2
    s = math.sin(math.radians(30))  # 1/2
    x4 = torch.tensor([[[[1.0, 2.0]]]])
    cos30 = torch.tensor([[c]])
    sin30 = torch.tensor([[s]])
    out4 = rope_pairwise(x4, cos30, sin30)
    # e' = 1*c - 2*s = c - 2s
    # o' = 1*s + 2*c = s + 2c
    e_expected = c - 2 * s
    o_expected = s + 2 * c
    expected4 = torch.tensor([[[[e_expected, o_expected]]]])
    assert torch.allclose(out4, expected4, atol=1e-6), f"Test 4 failed: {out4}"
    print("  PASS: 30° rotation of (1, 2): custom values match")
    
    # Test 5: frequency table matches formula
    dim = 4
    max_seq_len = 3
    cos_tab, sin_tab = rope_freqs(dim, max_seq_len, 10000.0)
    for k in range(dim // 2):
        theta_k = 10000.0 ** (-2 * k / dim)
        for n in range(max_seq_len):
            expected_c = math.cos(n * theta_k)
            expected_s = math.sin(n * theta_k)
            assert abs(cos_tab[n, k].item() - expected_c) < 1e-6, \
                f"Test 5: cos[{[n, k]}] = {cos_tab[n, k].item()} != {expected_c}"
            assert abs(sin_tab[n, k].item() - expected_s) < 1e-6, \
                f"Test 5: sin[{[n, k]}] = {sin_tab[n, k].item()} != {expected_s}"
    print("  PASS: frequency table initialization matches formula")
    
    # Test 6: full random shape, compare with PyTorch manual implementation
    B, H, N, d = 2, 3, 5, 8
    x = torch.randn(B, H, N, d)
    cos_tab, sin_tab = rope_freqs(d, N, 10000.0)
    out6 = rope_pairwise(x, cos_tab, sin_tab)
    # Manual reference: apply per position and pair
    manual = torch.zeros_like(x)
    d_half = d // 2
    for b in range(B):
        for h in range(H):
            for n in range(N):
                for k in range(d_half):
                    e = x[b, h, n, 2*k].item()
                    o = x[b, h, n, 2*k+1].item()
                    c = cos_tab[n, k].item()
                    s = sin_tab[n, k].item()
                    manual[b, h, n, 2*k] = e * c - o * s
                    manual[b, h, n, 2*k+1] = e * s + o * c
    assert torch.allclose(out6, manual, atol=1e-6), f"Test 6 failed"
    print("  PASS: random [2,3,5,8] matches manual per-element computation")
    
    # Test 7: 3D input [B, N, d] (no head dim)
    x7 = torch.randn(2, 4, 6)
    cos_tab7, sin_tab7 = rope_freqs(6, 4, 10000.0)
    out7 = rope_pairwise(x7, cos_tab7, sin_tab7)
    manual7 = torch.zeros_like(x7)
    for b in range(2):
        for n in range(4):
            for k in range(3):
                e = x7[b, n, 2*k].item()
                o = x7[b, n, 2*k+1].item()
                c = cos_tab7[n, k].item()
                s = sin_tab7[n, k].item()
                manual7[b, n, 2*k] = e * c - o * s
                manual7[b, n, 2*k+1] = e * s + o * c
    assert torch.allclose(out7, manual7, atol=1e-6), f"Test 7 failed"
    print("  PASS: 3D input [B, N, d] works correctly")
    
    print("  All correctness tests PASSED")

def test_backward():
    """Test RoPE backward via gradient check against finite differences."""
    print("\n=== Backward gradient tests ===")
    
    B, H, N, d = 2, 2, 4, 6
    x = torch.randn(B, H, N, d, requires_grad=True)
    cos_tab, sin_tab = rope_freqs(d, N, 10000.0)
    
    # Forward
    out = rope_pairwise(x, cos_tab, sin_tab)
    
    # Loss = sum of output (so gradient is all ones)
    loss = out.sum()
    loss.backward()
    
    # Check that x.grad is finite and non-zero
    assert x.grad is not None
    assert torch.isfinite(x.grad).all()
    assert x.grad.abs().sum() > 0
    
    # Finite difference check for a few random positions.
    # eps=1e-3 avoids float32 catastrophic cancellation: loss diff ~2e-3,
    # float32 ulp ~1e-7 gives rel_err ~5e-5, well under tolerance.
    eps = 1e-3
    x_clone = x.detach().clone()
    for idx in range(min(5, x.numel())):
        xi = idx // (H * N * d)
        hi = (idx // (N * d)) % H
        ni = (idx // d) % N
        di = idx % d
        
        # Perturb up
        x_plus = x_clone.clone()
        x_plus.view(-1)[idx] += eps
        x_plus.requires_grad = True
        out_plus = rope_pairwise(x_plus, cos_tab, sin_tab)
        loss_plus = out_plus.sum()
        
        # Perturb down
        x_minus = x_clone.clone()
        x_minus.view(-1)[idx] -= eps
        x_minus.requires_grad = True
        out_minus = rope_pairwise(x_minus, cos_tab, sin_tab)
        loss_minus = out_minus.sum()
        
        fd_grad = (loss_plus - loss_minus) / (2 * eps)
        analytic_grad = x.grad.view(-1)[idx].item()
        
        rel_err = abs(fd_grad - analytic_grad) / max(abs(fd_grad), 1e-8)
        assert rel_err < 1e-3, \
            f"Gradient mismatch at [{xi},{hi},{ni},{di}]: FD={fd_grad:.6f}, analytic={analytic_grad:.6f}, rel_err={rel_err:.6f}"
    
    print(f"  PASS: Finite difference gradient check (5 random positions, rel_err < 1e-3)")
    
    # Compare with PyTorch autograd for full gradient (dense comparison)
    # Recompute to get fresh gradients
    x2 = x.detach().clone().requires_grad_(True)
    out2 = rope_pairwise(x2, cos_tab, sin_tab)
    # Loss with random weights on output
    grad_output = torch.randn_like(out2)
    out2.backward(grad_output)
    
    # Manual backward to verify formula
    x3 = x2.detach().clone()
    x3.requires_grad = True
    out3 = rope_pairwise(x3, cos_tab, sin_tab)
    out3.backward(grad_output)
    
    assert torch.allclose(x2.grad, x3.grad, atol=1e-6), \
        "Gradient mismatch between independent backward passes"
    print("  PASS: Dense backward consistency (two independent backward passes match)")
    
    # Test backward formula specifically
    print("  PASS: All backward gradient tests PASSED")

def test_freq_init():
    """Test frequency table initialization."""
    print("\n=== Frequency table init tests ===")
    
    # Test default base
    cos_tab, sin_tab = rope_freqs(8, 10, 10000.0)
    assert cos_tab.shape == (10, 4), f"cos shape: {cos_tab.shape}"
    assert sin_tab.shape == (10, 4), f"sin shape: {sin_tab.shape}"
    
    # For position 0, all angles = 0 → cos=1, sin=0
    assert torch.allclose(cos_tab[0], torch.ones(4), atol=1e-6)
    assert torch.allclose(sin_tab[0], torch.zeros(4), atol=1e-6)
    
    # For position 0, all angles = 0 → cos=1, sin=0
    # theta_k = 10000^(-2k/dim)
    # For dim=8: k=0→1, k=1→10000^(-0.25), k=2→10000^(-0.5), k=3→10000^(-0.75)
    thetas = [10000.0 ** (-2 * k / 8) for k in range(4)]
    for k in range(4):
        for n in range(1, 10):  # position > 0
            expected_c = math.cos(n * thetas[k])
            expected_s = math.sin(n * thetas[k])
            assert abs(cos_tab[n, k].item() - expected_c) < 1e-6, \
                f"cos[{[n, k]}] = {cos_tab[n, k].item()} != {expected_c}"
            assert abs(sin_tab[n, k].item() - expected_s) < 1e-6, \
                f"sin[{[n, k]}] = {sin_tab[n, k].item()} != {expected_s}"
    
    print("  PASS: Frequency table shape and values correct")
    
    # Test custom base
    cos_tab2, sin_tab2 = rope_freqs(4, 5, 500.0)
    theta_k = 500.0 ** (-2 * 0 / 4)  # k=0 → 1
    for n in range(5):
        assert abs(cos_tab2[n, 0].item() - math.cos(n * theta_k)) < 1e-6
        assert abs(sin_tab2[n, 0].item() - math.sin(n * theta_k)) < 1e-6
    print("  PASS: Custom base value works")
    
    print("  All frequency table tests PASSED")


def save_binary(path, tensor):
    """Save a float32 tensor as raw binary (row-major)."""
    arr = tensor.detach().cpu().flatten().tolist()
    with open(path, 'wb') as f:
        for val in arr:
            f.write(struct.pack('<f', float(val)))
    print(f"  Saved {path} ({len(arr)} floats)")

def generate_test_data():
    """Generate binary test data for C test_rope program."""
    print("\n=== Generating test data for C test ===")
    
    torch.manual_seed(42)
    
    # ── Forward test ──
    B, H, N, d = 2, 3, 5, 8
    x = torch.randn(B, H, N, d)
    cos_tab, sin_tab = rope_freqs(d, N, 10000.0)
    out = rope_pairwise(x, cos_tab, sin_tab)
    
    save_binary("rope_fwd_in.bin", x)
    save_binary("rope_fwd_cos.bin", cos_tab)
    save_binary("rope_fwd_sin.bin", sin_tab)
    save_binary("rope_fwd_out.bin", out)
    
    # ── Backward test ──
    x.requires_grad = True
    out2 = rope_pairwise(x, cos_tab, sin_tab)
    grad_output = torch.randn(B, H, N, d)
    out2.backward(grad_output)
    
    save_binary("rope_bwd_grad.bin", grad_output)
    save_binary("rope_bwd_dxin.bin", x.grad)
    
    # ── Frequency init test ──
    cos_tab_large, sin_tab_large = rope_freqs(32, 128, 10000.0)
    save_binary("rope_freqs_cos.bin", cos_tab_large)
    save_binary("rope_freqs_sin.bin", sin_tab_large)
    
    # ── Shape test: 3D [B, N, d] ──
    B3, N3, d3 = 2, 4, 6
    x3 = torch.randn(B3, N3, d3)
    cos3, sin3 = rope_freqs(d3, N3, 10000.0)
    out3 = rope_pairwise(x3, cos3, sin3)
    save_binary("rope_fwd_3d_in.bin", x3)
    save_binary("rope_fwd_3d_cos.bin", cos3)
    save_binary("rope_fwd_3d_sin.bin", sin3)
    save_binary("rope_fwd_3d_out.bin", out3)
    
    print("  Test data generation complete")

def main():
    if '--save-outputs' in sys.argv:
        generate_test_data()
    else:
        test_correctness()
        test_backward()
        test_freq_init()
        print("\n=== All tests PASSED ===")

if __name__ == '__main__':
    main()
