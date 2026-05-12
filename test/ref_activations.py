# /// script
# requires-python = ">=3.10"
# dependencies = ["torch"]
# ///
"""PyTorch reference: activations (relu, softmax, cross_entropy).

Matches C test_autograd.c activation tests.
Usage: uv run test/ref_activations.py
"""

import torch

torch.manual_seed(42)
EPS = 1e-5


def check(name, got, expected):
    ok = torch.allclose(got, expected, atol=EPS)
    status = "OK" if ok else "FAIL"
    print(f"  {name}: max_diff={torch.max(torch.abs(got - expected)).item():.6e}  [{status}]")
    return ok


def test_relu_positive():
    print("  test_relu_positive...")
    a = torch.tensor([5.0], requires_grad=True)
    c = torch.relu(a)
    c.backward()
    check("da", a.grad, torch.tensor([1.0]))


def test_relu_negative():
    print("  test_relu_negative...")
    a = torch.tensor([-3.0], requires_grad=True)
    c = torch.relu(a)
    c.backward()
    check("da", a.grad, torch.tensor([0.0]))


def test_relu_mixed():
    print("  test_relu_mixed...")
    a = torch.tensor([-2.0, 0.0, 3.0], requires_grad=True)
    c = torch.relu(a)
    c.backward(torch.ones_like(c))
    check("da", a.grad, torch.tensor([0.0, 0.0, 1.0]))


def test_relu_chain():
    print("  test_relu_chain...")
    a = torch.tensor([-1.0, 2.0], requires_grad=True)
    b = torch.relu(a)   # [0, 2]
    c = b + a            # [-1, 4]
    c.backward(torch.ones_like(c))
    # dc/da0 = drelu/da0 + 1 = 0 + 1 = 1
    # dc/da1 = drelu/da1 + 1 = 1 + 1 = 2
    check("da", a.grad, torch.tensor([1.0, 2.0]))


def test_softmax_1d():
    print("  test_softmax_1d...")
    a = torch.tensor([1.0, 2.0, 3.0], requires_grad=True)
    c = torch.softmax(a, dim=0)
    loss = c.sum()
    loss.backward()
    # gradient of softmax sum is 1 for all inputs
    # sum(softmax)=1, so d(sum)/dx = 0
    check("da", a.grad, torch.tensor([0.0, 0.0, 0.0]))


def test_softmax_2d_dim0():
    print("  test_softmax_2d_dim0...")
    a = torch.tensor([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]], requires_grad=True)
    c = torch.softmax(a, dim=0)
    loss = c.sum()
    loss.backward()
    # sum(softmax)=1 per column, grad should be 0
    check("da", a.grad, torch.zeros_like(a.grad))


def test_softmax_2d_dim1():
    print("  test_softmax_2d_dim1...")
    a = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], requires_grad=True)
    c = torch.softmax(a, dim=1)
    loss = c.sum()
    loss.backward()
    check("da", a.grad, torch.zeros_like(a.grad))


def test_softmax_stability():
    print("  test_softmax_stability...")
    a = torch.tensor([1000.0, 1010.0, 1020.0], requires_grad=True)
    c = torch.softmax(a, dim=0)
    loss = c.sum()
    loss.backward()
    check("da", a.grad, torch.zeros_like(a.grad))


def test_softmax_sum_to_one():
    print("  test_softmax_sum_to_one...")
    a = torch.tensor([2.0, -1.0, 0.5], requires_grad=True)
    c = torch.softmax(a, dim=0)
    s = c.sum().item()
    print(f"    sum={s:.6f} (should be 1.0)")
    assert abs(s - 1.0) < EPS, f"softmax sum != 1: {s}"


def test_cross_entropy_simple():
    print("  test_cross_entropy_simple...")
    logits = torch.tensor([[1.0, 2.0, 3.0]], requires_grad=True)
    target = torch.tensor([2])
    loss = torch.nn.functional.cross_entropy(logits, target, reduction='mean')
    loss.backward()
    # expected: softmax([1,2,3]) = [0.0900, 0.2447, 0.6652]
    # grad: sm - one_hot = [0.0900, 0.2447, 0.6652-1]
    sm = torch.softmax(logits, dim=1)
    grad_expected = (sm - torch.nn.functional.one_hot(target, 3).float()) / 1.0
    check("dlogits", logits.grad, grad_expected)


def test_cross_entropy_batch():
    print("  test_cross_entropy_batch...")
    logits = torch.tensor([[1.0, 2.0, 3.0],
                            [4.0, 5.0, 6.0]], requires_grad=True)
    target = torch.tensor([2, 1])
    loss = torch.nn.functional.cross_entropy(logits, target, reduction='mean')
    loss.backward()
    # grad = (softmax - one_hot) / batch_size
    sm = torch.softmax(logits, dim=1)
    grad_expected = (sm - torch.nn.functional.one_hot(target, 3).float()) / 2.0
    check("dlogits", logits.grad, grad_expected)


def test_cross_entropy_stability():
    print("  test_cross_entropy_stability...")
    logits = torch.tensor([[1000.0, 1010.0, 1020.0]], requires_grad=True)
    target = torch.tensor([2])
    loss = torch.nn.functional.cross_entropy(logits, target, reduction='mean')
    loss.backward()
    sm = torch.softmax(logits, dim=1)
    grad_expected = (sm - torch.nn.functional.one_hot(target, 3).float()) / 1.0
    check("dlogits", logits.grad, grad_expected)


if __name__ == "__main__":
    print("=== PyTorch Reference: Activations ===")
    tests = [
        ("relu_positive", test_relu_positive),
        ("relu_negative", test_relu_negative),
        ("relu_mixed", test_relu_mixed),
        ("relu_chain", test_relu_chain),
        ("softmax_1d", test_softmax_1d),
        ("softmax_2d_dim0", test_softmax_2d_dim0),
        ("softmax_2d_dim1", test_softmax_2d_dim1),
        ("softmax_stability", test_softmax_stability),
        ("softmax_sum_to_one", test_softmax_sum_to_one),
        ("cross_entropy_simple", test_cross_entropy_simple),
        ("cross_entropy_batch", test_cross_entropy_batch),
        ("cross_entropy_stability", test_cross_entropy_stability),
    ]
    for name, fn in tests:
        fn()
    print("  ALL PASS")
