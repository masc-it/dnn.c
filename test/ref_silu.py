#!/usr/bin/env python3
"""PyTorch reference for SiLU (Swish) activation: x * sigmoid(x).

Usage:
    python3 test/ref_silu.py          # print reference values

Output: float arrays of forward values and gradients for known inputs,
matching expected precision (EPS=1e-5) in the C tests.
"""

import torch
import torch.nn.functional as F
import math

torch.manual_seed(42)

# ── Scalar cases ──
print("=== Scalar ===")
for x_val in [0.0, 1.0, -1.0, 2.0, -2.0, 3.0, -3.0, 0.5, -0.5]:
    x = torch.tensor([x_val], dtype=torch.float32, requires_grad=True)
    y = F.silu(x)
    y.backward()
    print(f"  x={x_val:5.1f}  silu={y.item():.8f}  grad={x.grad.item():.8f}")

# ── Array case ──
print("=== Array ===")
vals = [-3.0, -1.0, 0.0, 2.0, 4.0]
x = torch.tensor(vals, dtype=torch.float32, requires_grad=True)
y = F.silu(x)
grad_output = torch.ones_like(y)
y.backward(grad_output)
print("  x:", [f"{v:.1f}" for v in vals])
print("  silu:", [f"{v:.8f}" for v in y.detach().tolist()])
print("  grad:", [f"{v:.8f}" for v in x.grad.tolist()])

# ── 2D tensor case ──
print("=== 2D Tensor [2,3] ===")
x2d = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype=torch.float32, requires_grad=True)
y2d = F.silu(x2d)
grad_out = torch.ones_like(y2d)
y2d.backward(grad_out)
print("  silu:")
for row in y2d.detach().tolist():
    print("   ", [f"{v:.8f}" for v in row])
print("  grad:")
for row in x2d.grad.tolist():
    print("   ", [f"{v:.8f}" for v in row])

# ── Non-contiguous (transposed slice) ──
print("=== 2D non-contiguous (transposed) [3,2] ===")
x_nc = torch.tensor([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]], dtype=torch.float32, requires_grad=True)
x_t = x_nc.T  # [2,3], non-contiguous in PyTorch terms
# silu on transposed
y_nc = F.silu(x_t)
grad_out = torch.ones_like(y_nc)
y_nc.backward(grad_out)
print("  silu:")
for row in y_nc.detach().tolist():
    print("   ", [f"{v:.8f}" for v in row])
print("  x.grad (accumulated on original):")
for row in x_nc.grad.tolist():
    print("   ", [f"{v:.8f}" for v in row])

print("\nDone.")
