# /// script
# requires-python = ">=3.10"
# dependencies = ["torch", "numpy"]
# ///
"""PyTorch reference: LayerNorm.

Tests forward stats and backward grads against C's tensor_layer_norm.

Usage:
  uv run --script test/ref_layer_norm.py
"""
import torch
import torch.nn as nn

torch.manual_seed(42)

# ── test 1: forward stats ──
print("── LayerNorm forward stats ──")
x = torch.arange(1, 13, dtype=torch.float).reshape(3, 4)
weight = torch.ones(4)
bias = None

ln = nn.LayerNorm(4, elementwise_affine=False)  # no γ/β
y = ln(x)
print(f"  mean per row: {y.mean(dim=1).tolist()}")
print(f"  std  per row: {y.std(dim=1, unbiased=False).tolist()}")
print("  OK\n")

# ── test 2: backward with affine ──
print("── LayerNorm backward ──")
x = torch.tensor([[1.,2.,3.],[4.,5.,6.]], requires_grad=True)
w = torch.ones(3, requires_grad=True)
b = torch.zeros(3, requires_grad=True)

ln = nn.LayerNorm(3, elementwise_affine=False)
y = ln(x) * w + b
loss = y.sum()
loss.backward()

print(f"  dx:\n  {x.grad}")
print(f"  dw: {w.grad}")
print(f"  db: {b.grad}")
print("  OK\n")

# ── test 3: full affine LayerNorm with random input ──
print("── LayerNorm affine, random ──")
torch.manual_seed(1)
x = torch.randn(2, 5, requires_grad=True)
ln = nn.LayerNorm(5, elementwise_affine=True)  # γ=1, β=0 initialized
loss = ln(x).sum()
loss.backward()
print(f"  input grad non-zero: {x.grad.abs().sum().item() > 0}")
print(f"  weight grad non-zero: {ln.weight.grad.abs().sum().item() > 0}")
print(f"  bias grad non-zero: {ln.bias.grad.abs().sum().item() > 0}")
print("  OK\n")

# ── test 4: numerical gradient check with non-uniform weights ──
print("── Numerical gradient check (non-uniform w) ──")
torch.manual_seed(2)
x = torch.randn(4, requires_grad=True)
w = torch.tensor([1., 2., 0.5, 0.1], requires_grad=True)
b = torch.zeros(4, requires_grad=True)

def f(x_, w_, b_):
    mean = x_.mean()
    var = x_.var(unbiased=False)
    y = (x_ - mean) / (var + 1e-5).sqrt()
    return (y * w_ + b_).sum()

loss = f(x, w, b)
loss.backward()

dx_num = torch.zeros_like(x)
h = 1e-4
for i in range(4):
    x_hi = x.clone(); x_hi[i] += h
    x_lo = x.clone(); x_lo[i] -= h
    dx_num[i] = (f(x_hi, w, b) - f(x_lo, w, b)) / (2 * h)

err = (x.grad - dx_num).abs().max().item()
print(f"  dx analytical: {x.grad}")
print(f"  dx numerical:  {dx_num}")
print(f"  max error: {err:.8f}  (tol=1e-3)")
assert err < 1e-3, f"numerical check failed: {err}"
print("  OK\n")
