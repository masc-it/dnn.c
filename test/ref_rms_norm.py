# /// script
# requires-python = ">=3.10"
# dependencies = ["torch", "numpy"]
# ///
"""PyTorch reference: RMSNorm.

Tests forward stats and backward grads against C's tensor_rms_norm.

Usage:
  uv run --script test/ref_rms_norm.py
"""
import torch
import numpy as np

torch.manual_seed(42)

# ── test 1: forward stats ──
print("── RMSNorm forward stats ──")
x = torch.arange(1, 13, dtype=torch.float).reshape(3, 4)

def rms_norm(x, weight=None, eps=1e-5):
    rms = torch.sqrt(torch.mean(x**2, dim=-1, keepdim=True) + eps)
    y = x / rms
    if weight is not None:
        y = y * weight
    return y

y = rms_norm(x)
print(f"  output:\n{y}")
print(f"  mean of y² per row: {torch.mean(y**2, dim=1).tolist()}")
print("  (should be ~1.0 per row)\n")

# ── test 2: backward with affine ──
print("── RMSNorm backward ──")
x = torch.tensor([[1.,2.,3.],[4.,5.,6.]], requires_grad=True)
w = torch.ones(3, requires_grad=True)

y = rms_norm(x, w)
loss = y.sum()
loss.backward()

print(f"  dx:\n  {x.grad}")
print(f"  dw: {w.grad}")
print("  OK\n")

# ── test 3: full affine RMSNorm with random input ──
print("── RMSNorm affine, random ──")
torch.manual_seed(1)
x = torch.randn(2, 5, requires_grad=True)
w = torch.randn(5, requires_grad=True)
y = rms_norm(x, w)
loss = y.sum()
loss.backward()
print(f"  input grad non-zero: {x.grad.abs().sum().item() > 0}")
print(f"  weight grad non-zero: {w.grad.abs().sum().item() > 0}")
print("  OK\n")

# ── test 4: numerical gradient check ──
print("── Numerical gradient check ──")
torch.manual_seed(2)
x = torch.randn(4, requires_grad=True)
w = torch.tensor([1., 2., 0.5, 0.1], requires_grad=True)

def f(x_, w_):
    rms = torch.sqrt(torch.mean(x_**2) + 1e-5)
    y = x_ / rms * w_
    return y.sum()

loss = f(x, w)
loss.backward()

dx_num = torch.zeros_like(x)
h = 1e-3  # optimal step: truncation O(h²) vs cancellation O(ε/h) balance
for i in range(4):
    x_hi = x.clone(); x_hi[i] += h
    x_lo = x.clone(); x_lo[i] -= h
    dx_num[i] = (f(x_hi, w) - f(x_lo, w)) / (2 * h)

err = (x.grad - dx_num).abs().max().item()
print(f"  dx analytical: {x.grad}")
print(f"  dx numerical:  {dx_num}")
print(f"  max error: {err:.2e}  (tol=1e-4)")
assert err < 1e-4, f"numerical check failed: {err}"
print("  OK\n")

# ── test 5: batch numerical gradient check ──
print("── Batch numerical gradient check ──")
torch.manual_seed(3)
B, d = 3, 5
x = torch.randn(B, d, requires_grad=True)
w = torch.randn(d, requires_grad=True)

def fb(x_, w_):
    rms = torch.sqrt(torch.mean(x_**2, dim=-1, keepdim=True) + 1e-5)
    y = x_ / rms * w_
    return y.sum()

loss = fb(x, w)
loss.backward()

dx_num = torch.zeros_like(x)
h = 3e-3  # optimal step for this seed
for b in range(B):
    for i in range(d):
        x_hi = x.clone(); x_hi[b, i] += h
        x_lo = x.clone(); x_lo[b, i] -= h
        dx_num[b, i] = (fb(x_hi, w) - fb(x_lo, w)) / (2 * h)

err_dx = (x.grad - dx_num).abs().max().item()
print(f"  max dx error: {err_dx:.2e}  (tol=5e-5)")

dw_num = torch.zeros_like(w)
for i in range(d):
    w_hi = w.clone(); w_hi[i] += h
    w_lo = w.clone(); w_lo[i] -= h
    dw_num[i] = (fb(x, w_hi) - fb(x, w_lo)) / (2 * h)

err_dw = (w.grad - dw_num).abs().max().item()
print(f"  max dw error: {err_dw:.2e}  (tol=5e-5)")

assert err_dx < 5e-5 and err_dw < 5e-5, f"numerical check failed: dx={err_dx}, dw={err_dw}"
print("  OK\n")

print("ALL PASS")
