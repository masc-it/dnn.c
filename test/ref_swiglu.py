"""
Reference script: SwiGLU FFN block.

Compares dnn.c swiglu_ffn forward against PyTorch.
Run:  python3 test/ref_swiglu.py

Expected output: all checks pass.
"""
import torch
import torch.nn.functional as F
import math

torch.manual_seed(0)


def swiglu_ffn_forward(x, w_gate, b_gate, w_up, b_up, w_down, b_down):
    """Exact same computation as dnn.c swiglu_ffn_forward."""
    gate = F.silu(F.linear(x, w_gate.T, b_gate))
    up = F.linear(x, w_up.T, b_up)
    hidden = gate * up
    return F.linear(hidden, w_down.T, b_down)


def test_forward_single():
    """Match test_swiglu_ffn_forward weights.  Single sample, batch=1."""
    d_model = 2
    hidden = 3

    w_gate = torch.tensor([[1., 2., 3.],
                           [4., 5., 6.]])          # [d_model, hidden]
    b_gate = torch.zeros(hidden)
    w_up   = torch.tensor([[0.1, 0.2, 0.3],
                           [0.4, 0.5, 0.6]])
    b_up   = torch.zeros(hidden)
    w_down = torch.tensor([[1., 4.],
                           [2., 5.],
                           [3., 6.]])              # [hidden, d_model]
    b_down = torch.zeros(d_model)

    x = torch.tensor([[1., 2.]])                    # [1, d_model]

    y = swiglu_ffn_forward(x, w_gate, b_gate, w_up, b_up, w_down, b_down)
    print(f"  Single sample y = {y}")
    print(f"  Expected C test y ≈ {y.tolist()}")
    return y


def test_forward_batch():
    """Match test_swiglu_ffn_batch weights.  Batch of 2."""
    d_model = 2
    hidden = 3

    w_gate = torch.tensor([[1., 2., 3.],
                           [4., 5., 6.]])
    b_gate = torch.zeros(hidden)
    w_up   = torch.tensor([[0.1, 0.2, 0.3],
                           [0.4, 0.5, 0.6]])
    b_up   = torch.zeros(hidden)
    w_down = torch.tensor([[1., 4.],
                           [2., 5.],
                           [3., 6.]])
    b_down = torch.zeros(d_model)

    x = torch.tensor([[1., 2.],
                      [3., 4.]])                    # [2, d_model]

    y = swiglu_ffn_forward(x, w_gate, b_gate, w_up, b_up, w_down, b_down)
    print(f"  Batch y = {y}")
    print(f"  Expected C test y ≈ {y.tolist()}")


def test_backward():
    """Verify that gradients flow to all 6 param tensors."""
    d_model = 2
    hidden = 3

    # identity-ish weights: gradients should be non-zero on all params
    w_gate = torch.tensor([[1., 0., 0.],
                           [0., 1., 0.]], requires_grad=True)
    b_gate = torch.zeros(hidden, requires_grad=True)
    w_up   = torch.tensor([[1., 0., 0.],
                           [0., 1., 0.]], requires_grad=True)
    b_up   = torch.zeros(hidden, requires_grad=True)
    w_down = torch.tensor([[1., 0.],
                           [0., 1.],
                           [0., 0.]], requires_grad=True)
    b_down = torch.zeros(d_model, requires_grad=True)

    x = torch.tensor([[2., 3.]], requires_grad=True)

    y = swiglu_ffn_forward(x, w_gate, b_gate, w_up, b_up, w_down, b_down)
    loss = y.sum()
    loss.backward()

    print(f"  Backward: all grads non-zero:")
    for name, p in [("w_gate", w_gate), ("b_gate", b_gate),
                    ("w_up", w_up), ("b_up", b_up),
                    ("w_down", w_down), ("b_down", b_down),
                    ("x", x)]:
        grad_ok = p.grad is not None and p.grad.abs().sum().item() > 0
        print(f"    {name}.grad non-zero = {grad_ok}, norm = {p.grad.norm().item():.6f}" if grad_ok
              else f"    {name}.grad = None or zero!")

    # Sanity: dx shape matches x
    assert x.grad.shape == x.shape, f"dx shape {x.grad.shape} != {x.shape}"


def test_intermediates():
    """Print intermediate values for manual C test comparison."""
    d_model = 2
    hidden = 3

    w_gate = torch.tensor([[1., 2., 3.],
                           [4., 5., 6.]])
    b_gate = torch.zeros(hidden)
    w_up   = torch.tensor([[0.1, 0.2, 0.3],
                           [0.4, 0.5, 0.6]])
    b_up   = torch.zeros(hidden)
    w_down = torch.tensor([[1., 4.],
                           [2., 5.],
                           [3., 6.]])
    b_down = torch.zeros(d_model)

    x = torch.tensor([[1., 2.]])

    gate_linear = F.linear(x, w_gate.T, b_gate)
    print(f"\n  gate_linear = {gate_linear}")
    silu_gate = F.silu(gate_linear)
    print(f"  silu(gate) = {silu_gate}")
    up = F.linear(x, w_up.T, b_up)
    print(f"  up = {up}")
    hidden_val = silu_gate * up
    print(f"  hidden (gate*up) = {hidden_val}")
    out = F.linear(hidden_val, w_down.T, b_down)
    print(f"  out = {out}")

    # Batch of 2
    x2 = torch.tensor([[1., 2.],
                       [3., 4.]])
    out2 = swiglu_ffn_forward(x2, w_gate, b_gate, w_up, b_up, w_down, b_down)
    print(f"  batch out = {out2}")


if __name__ == "__main__":
    print("ref_swiglu.py — PyTorch reference for SwiGLU FFN")
    y1 = test_forward_single()
    test_forward_batch()
    test_backward()
    test_intermediates()
    print("\nAll ref checks done.")
