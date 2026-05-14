# /// script
# requires-python = ">=3.10"
# dependencies = ["torch"]
# ///
"""PyTorch reference: sigmoid forward + backward.

Matches C test_autograd.c sigmoid tests.
Usage: uv run test/ref_sigmoid.py
"""

import torch

torch.manual_seed(42)
EPS = 1e-5


def check(name, got, expected):
    ok = torch.allclose(got, expected, atol=EPS)
    status = "OK" if ok else "FAIL"
    print(f"  {name}: max_diff={torch.max(torch.abs(got - expected)).item():.6e}  [{status}]")
    return ok


def test_sigmoid_simple():
    print("  test_sigmoid_simple...")
    a = torch.tensor([0.0], requires_grad=True)
    c = torch.sigmoid(a)
    assert abs(c.item() - 0.5) < EPS, f"sigmoid(0) != 0.5: {c.item()}"
    c.backward()
    # dσ/dx at 0 = 0.5 * 0.5 = 0.25
    check("da", a.grad, torch.tensor([0.25]))


def test_sigmoid_positive():
    print("  test_sigmoid_positive...")
    a = torch.tensor([2.0], requires_grad=True)
    c = torch.sigmoid(a)
    sig_ref = 1.0 / (1.0 + torch.exp(-a))
    assert abs(c.item() - sig_ref.item()) < EPS, f"sigmoid(2) mismatch"
    c.backward()
    expected_grad = sig_ref * (1.0 - sig_ref)
    check("da", a.grad, expected_grad)


def test_sigmoid_negative():
    print("  test_sigmoid_negative...")
    a = torch.tensor([-2.0], requires_grad=True)
    c = torch.sigmoid(a)
    sig_ref = 1.0 / (1.0 + torch.exp(-a))
    assert abs(c.item() - sig_ref.item()) < EPS, f"sigmoid(-2) mismatch"
    c.backward()
    expected_grad = sig_ref * (1.0 - sig_ref)
    check("da", a.grad, expected_grad)


def test_sigmoid_array():
    print("  test_sigmoid_array...")
    vals = torch.tensor([-3.0, -1.0, 0.0, 2.0, 4.0])
    a = vals.clone().requires_grad_(True)
    c = torch.sigmoid(a)

    # forward
    expected = 1.0 / (1.0 + torch.exp(-vals))
    check("forward", c, expected)

    # backward
    c.backward(torch.ones_like(c))
    expected_grad = expected * (1.0 - expected)
    check("backward", a.grad, expected_grad)


def test_sigmoid_chain():
    print("  test_sigmoid_chain...")
    # c = sigmoid(a) + a
    # dc/da = σ(a)*(1-σ(a)) + 1
    a = torch.tensor([1.0], requires_grad=True)
    c = torch.sigmoid(a) + a
    c.backward()
    sig = torch.sigmoid(torch.tensor([1.0]))
    expected_grad = sig * (1.0 - sig) + 1.0
    check("da", a.grad, expected_grad)


def test_sigmoid_no_grad():
    print("  test_sigmoid_no_grad...")
    a = torch.tensor([3.0])  # no requires_grad
    c = torch.sigmoid(a)
    sig_ref = 1.0 / (1.0 + torch.exp(-torch.tensor([3.0])))
    check("forward", c, sig_ref)
    assert a.grad is None, "no-grad: a.grad should be None"
    print("  OK (no grad allocated)")


if __name__ == "__main__":
    print("=== PyTorch Reference: Sigmoid ===")
    tests = [
        ("sigmoid_simple", test_sigmoid_simple),
        ("sigmoid_positive", test_sigmoid_positive),
        ("sigmoid_negative", test_sigmoid_negative),
        ("sigmoid_array", test_sigmoid_array),
        ("sigmoid_chain", test_sigmoid_chain),
        ("sigmoid_no_grad", test_sigmoid_no_grad),
    ]
    for name, fn in tests:
        fn()
    print("  ALL PASS")
