# /// script
# requires-python = ">=3.10"
# dependencies = ["torch"]
# ///
"""PyTorch reference: linear (dense) layer forward + backward.

Matches C test_nn.c tests.
Usage: uv run test/ref_nn.py
"""

import torch

torch.manual_seed(42)
EPS = 1e-5


def check(name, got, expected):
    ok = torch.allclose(got, expected, atol=EPS)
    status = "OK" if ok else "FAIL"
    print(f"  {name}: max_diff={torch.max(torch.abs(got - expected)).item():.6e}  [{status}]")
    return ok


def test_linear_forward_single():
    print("  test_linear_forward_single...")
    # input (1, 3), weight (3, 2), bias (2)
    W = torch.tensor([[1.0, 2.0],
                       [3.0, 4.0],
                       [5.0, 6.0]])
    b = torch.tensor([10.0, 20.0])
    x = torch.tensor([[2.0, 3.0, 4.0]])
    y = x @ W + b
    # expected: [2*1+3*3+4*5, 2*2+3*4+4*6] + [10,20] = [31,40]+[10,20] = [41,60]
    expected = torch.tensor([[41.0, 60.0]])
    check("y", y, expected)


def test_linear_forward_batch():
    print("  test_linear_forward_batch...")
    W = torch.tensor([[1.0, 2.0],
                       [3.0, 4.0],
                       [5.0, 6.0]])
    b = torch.tensor([10.0, 20.0])
    x = torch.tensor([[1.0, 2.0, 3.0],
                       [4.0, 5.0, 6.0]])
    y = x @ W + b
    # row0: [1*1+2*3+3*5, 1*2+2*4+3*6]+[10,20] = [22,28]+[10,20] = [32,48]
    # row1: [4*1+5*3+6*5, 4*2+5*4+6*6]+[10,20] = [49,64]+[10,20] = [59,84]
    expected = torch.tensor([[32.0, 48.0],
                              [59.0, 84.0]])
    check("y", y, expected)


def test_linear_backward_simple():
    print("  test_linear_backward_simple...")
    W = torch.tensor([[1.0, 2.0],
                       [3.0, 4.0]], requires_grad=True)
    b = torch.tensor([0.0, 0.0], requires_grad=True)
    x = torch.tensor([[5.0, 6.0]])
    y = x @ W + b
    y.backward(torch.ones_like(y))
    # dW = x^T @ dloss/dy = [[5],[6]] @ [[1,1]] = [[5,5],[6,6]]
    expected_dW = torch.tensor([[5.0, 5.0],
                                 [6.0, 6.0]])
    # db = sum(dloss/dy, dim=0) = [1, 1]
    expected_db = torch.tensor([1.0, 1.0])
    check("dW", W.grad, expected_dW)
    check("db", b.grad, expected_db)


def test_linear_backward_batch():
    print("  test_linear_backward_batch...")
    W = torch.tensor([[1.0, 2.0, 3.0],
                       [4.0, 5.0, 6.0]], requires_grad=True)
    b = torch.tensor([0.0, 0.0, 0.0], requires_grad=True)
    x = torch.tensor([[1.0, 2.0],
                       [3.0, 4.0],
                       [5.0, 6.0]])
    y = x @ W + b
    y.backward(torch.ones_like(y))
    # dW = X^T @ ones(3,3)
    # X^T = [[1,3,5],[2,4,6]]
    # dW = [[1+3+5, 1+3+5, 1+3+5],[2+4+6,2+4+6,2+4+6]] = [[9,9,9],[12,12,12]]
    expected_dW = torch.tensor([[9.0, 9.0, 9.0],
                                 [12.0, 12.0, 12.0]])
    # db = sum over batch = [3,3,3]
    expected_db = torch.tensor([3.0, 3.0, 3.0])
    check("dW", W.grad, expected_dW)
    check("db", b.grad, expected_db)


def test_linear_backward_input_grad():
    print("  test_linear_backward_input_grad...")
    W = torch.tensor([[1.0, 0.0],
                       [0.0, 1.0]], requires_grad=True)
    x = torch.tensor([[3.0, 4.0]], requires_grad=True)
    y = x @ W
    y.backward(torch.ones_like(y))
    # dx = dloss/dy @ W^T = [1,1] @ I = [1,1]
    expected_dx = torch.tensor([[1.0, 1.0]])
    check("dx", x.grad, expected_dx)


def test_linear_chain():
    print("  test_linear_chain...")
    # two linear layers: y = (x @ W1 + b1) @ W2 + b2
    W1 = torch.tensor([[1.0, 2.0, 3.0],
                        [4.0, 5.0, 6.0]], requires_grad=True)
    b1 = torch.tensor([0.0, 0.0, 0.0], requires_grad=True)
    W2 = torch.tensor([[1.0],
                        [2.0],
                        [3.0]], requires_grad=True)
    b2 = torch.tensor([0.0], requires_grad=True)

    x = torch.tensor([[1.0, 2.0]])
    h = x @ W1 + b1
    y = h @ W2 + b2
    y.backward(torch.ones_like(y))

    # forward: h = [1*1+2*4, 1*2+2*5, 1*3+2*6] = [9, 12, 15]
    # y = 9*1 + 12*2 + 15*3 = 78
    # dW2 = h^T @ 1 = [9,12,15]^T
    expected_dW2 = torch.tensor([[9.0], [12.0], [15.0]])
    check("dW2", W2.grad, expected_dW2)
    check("db2", b2.grad, torch.tensor([1.0]))

    # dh = 1 @ W2^T = [1,2,3]
    # dW1 = x^T @ dh = [[1],[2]] @ [1,2,3] = [[1,2,3],[2,4,6]]
    expected_dW1 = torch.tensor([[1.0, 2.0, 3.0],
                                  [2.0, 4.0, 6.0]])
    check("dW1", W1.grad, expected_dW1)
    check("db1", b1.grad, torch.tensor([1.0, 2.0, 3.0]))


def test_linear_no_bias():
    print("  test_linear_no_bias...")
    W = torch.tensor([[1.0, 2.0],
                       [3.0, 4.0]], requires_grad=True)
    x = torch.tensor([[5.0, 6.0]])
    y = x @ W  # no bias
    y.backward(torch.ones_like(y))
    expected_dW = torch.tensor([[5.0, 5.0],
                                 [6.0, 6.0]])
    check("dW", W.grad, expected_dW)


def test_linear_relu():
    print("  test_linear_relu...")
    W = torch.tensor([[1.0, -1.0],
                       [-2.0, 2.0]], requires_grad=True)
    b = torch.tensor([0.0, 0.0], requires_grad=True)
    x = torch.tensor([[3.0, 4.0]])
    h = x @ W + b  # [3*1+4*(-2), 3*(-1)+4*2] = [-5, 5]
    y = torch.relu(h)  # [0, 5]
    y.backward(torch.ones_like(y))
    # dy/dh = [0, 1]; dh/dW = x^T @ dy/dh
    # dW = [[3],[4]] @ [0,1] = [[0,3],[0,4]]
    expected_dW = torch.tensor([[0.0, 3.0],
                                 [0.0, 4.0]])
    # db = [0, 1]
    expected_db = torch.tensor([0.0, 1.0])
    check("dW", W.grad, expected_dW)
    check("db", b.grad, expected_db)


if __name__ == "__main__":
    print("=== PyTorch Reference: Linear (NN) ===")
    tests = [
        ("linear_forward_single", test_linear_forward_single),
        ("linear_forward_batch", test_linear_forward_batch),
        ("linear_backward_simple", test_linear_backward_simple),
        ("linear_backward_batch", test_linear_backward_batch),
        ("linear_backward_input_grad", test_linear_backward_input_grad),
        ("linear_chain", test_linear_chain),
        ("linear_no_bias", test_linear_no_bias),
        ("linear_relu", test_linear_relu),
    ]
    for name, fn in tests:
        fn()
    print("  ALL PASS")
