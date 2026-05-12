# /// script
# requires-python = ">=3.10"
# dependencies = ["torch", "numpy"]
# ///
"""Conv2D reference values for dnn.c cross-validation."""
import torch
import numpy as np

torch.manual_seed(0)

# ── test case 1: 1x1 kernel (just a matmul per pixel) ──
print("── test 1: 1x1 kernel, no padding ──")
N, C, H, W, out_C = 1, 2, 3, 3, 4
x = torch.randn(N, C, H, W, requires_grad=True)
w = torch.randn(out_C, C, 1, 1, requires_grad=True)
b = torch.randn(out_C, requires_grad=True)

y = torch.nn.functional.conv2d(x, w, b, stride=1, padding=0)
loss = y.sum()
loss.backward()

print(f"  x shape: {x.shape}  w shape: {w.shape}")
print(f"  out shape: {y.shape}")
print(f"  out[0,0,0,0] = {y[0,0,0,0].item():.8f}")
print(f"  dx shape: {x.grad.shape}")
print(f"  dw[0,0,0,0] = {w.grad[0,0,0,0].item():.8f}")
print(f"  db[0] = {b.grad[0].item():.8f}")

# save values for C test
print(f"  x_data: {x.detach().flatten().tolist()}")
print(f"  w_data: {w.detach().flatten().tolist()}")
print(f"  b_data: {b.detach().flatten().tolist()}")
print(f"  out_data: {y.detach().flatten().tolist()}")
print(f"  dx_data: {x.grad.flatten().tolist()}")
print(f"  dw_data: {w.grad.flatten().tolist()}")
print(f"  db_data: {b.grad.flatten().tolist()}")
print()

# ── test case 2: 2x2 kernel, stride 2 (downsample) ──
print("── test 2: 2x2 kernel, stride 2 ──")
x2 = torch.tensor([[[[1.,2.,3.,4.],[5.,6.,7.,8.],
                      [9.,10.,11.,12.],[13.,14.,15.,16.]]]],
                   requires_grad=True)  # (1,1,4,4)
w2 = torch.tensor([[[[1.,0.],[0.,1.]]]], requires_grad=True)  # (1,1,2,2)
b2 = torch.tensor([0.], requires_grad=True)

y2 = torch.nn.functional.conv2d(x2, w2, b2, stride=2, padding=0)
loss2 = y2.sum()
loss2.backward()

print(f"  out: {y2.flatten().tolist()}")
print(f"  dx:  {x2.grad.flatten().tolist()}")
print(f"  dw:  {w2.grad.flatten().tolist()}")
print(f"  db:  {b2.grad.flatten().tolist()}")
print()

# ── test case 3: 3x3 kernel, padding=1 (same output size) ──
print("── test 3: 3x3 kernel, pad=1 (same-spatial) ──")
x3 = torch.randn(2, 3, 4, 4, requires_grad=True)
w3 = torch.randn(2, 3, 3, 3, requires_grad=True)
b3 = torch.randn(2, requires_grad=True)

y3 = torch.nn.functional.conv2d(x3, w3, b3, stride=1, padding=1)
loss3 = y3.sum()
loss3.backward()

print(f"  out shape: {y3.shape}  (expected 2,2,4,4)")
print(f"  all non-zero grads: "
      f"dx={x3.grad.abs().sum().item():.4f} "
      f"dw={w3.grad.abs().sum().item():.4f} "
      f"db={b3.grad.abs().sum().item():.4f}")
print(f"  dx_data: {x3.grad.flatten().tolist()}")
print(f"  dw_data: {w3.grad.flatten().tolist()}")
print(f"  db_data: {b3.grad.flatten().tolist()}")

# ── test case 4: 2x2 kernel, no padding, stride 1 (simple manual) ──
print("\n── test 4: 2x2 kernel, stride=1, pad=0 (manual verify) ──")
x4 = torch.tensor([[[[1.,2.],[3.,4.]]]], requires_grad=True)
w4 = torch.tensor([[[[1.,0.],[0.,1.]]]], requires_grad=True)
b4 = torch.tensor([0.], requires_grad=True)
y4 = torch.nn.functional.conv2d(x4, w4, b4, stride=1, padding=0)
loss4 = y4.sum()
loss4.backward()
# expected: out=[[[[5]]]], dx=[[[[1,0],[0,1]]]], dw=[[[[1,2],[3,4]]]]
print(f"  out: {y4.flatten().tolist()}")
print(f"  dx:  {x4.grad.flatten().tolist()}")
print(f"  dw:  {w4.grad.flatten().tolist()}")
print(f"  db:  {b4.grad.flatten().tolist()}")
# verification
assert abs(y4[0,0,0,0].item() - 5.0) < 1e-6, "out should be 5"
assert abs(w4.grad[0,0,0,0].item() - 1.0) < 1e-6, "dw[0,0,0,0] should be 1"
assert abs(w4.grad[0,0,0,1].item() - 2.0) < 1e-6, "dw[0,0,0,1] should be 2"
assert abs(w4.grad[0,0,1,0].item() - 3.0) < 1e-6, "dw[0,0,1,0] should be 3"
assert abs(w4.grad[0,0,1,1].item() - 4.0) < 1e-6, "dw[0,0,1,1] should be 4"
assert abs(b4.grad[0].item() - 1.0) < 1e-6, "db should be 1"
print("  manual verification PASS")
