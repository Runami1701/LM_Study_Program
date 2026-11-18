"""Checks for a-l03 — backward(): topological sort."""

import numpy as np
import pytest

from tests.helpers import assert_close, load


def T():
    return load("autograd.tensor").Tensor


@pytest.mark.fn("backward")
def test_simple_chain():
    Tn = T()
    x = Tn(np.array(3.0))
    y = x * 2.0 + 1.0
    z = y * y
    z.backward()
    assert_close(x.grad, 28.0, "dz/dx for z = (2x + 1)^2 at x = 3", hint="dz/dx = 2(2x+1)·2 = 28")


@pytest.mark.fn("backward")
def test_diamond():
    Tn = T()
    a = Tn(np.array(2.0))
    b = a * 2.0
    c = a * 3.0
    d = b * c                   # d = 6 a^2  ->  dd/da = 12 a = 24
    d.backward()
    assert_close(a.grad, 24.0, "a.grad in the diamond d = (2a)(3a)",
                 hint="a reaches d by two paths; b and c must both finish before a's _backward runs")


@pytest.mark.fn("backward")
def test_deep_chain_no_recursion_limit():
    Tn = T()
    x = Tn(np.array(1.0))
    z = x
    for _ in range(5000):
        z = z * 1.0 + 0.0
    try:
        z.backward()
    except RecursionError:
        raise AssertionError("RecursionError on a 10,000-node chain — build the topological order with an explicit stack, not recursion")
    assert_close(x.grad, 1.0, "x.grad after a 10,000-node chain")


@pytest.mark.fn("backward")
def test_root_seeded_and_backward_twice_accumulates_leaves():
    Tn = T()
    w = Tn(np.array([1.0, 2.0]))
    h = w * w
    loss = load("autograd.ops").sum(h) if hasattr(load("autograd.ops"), "sum") else None
    if loss is None:
        pytest.skip("needs ops.sum from a-l02")
    loss.backward()
    assert_close(loss.grad, 1.0, "loss.grad", hint="backward seeds the root with dL/dL = 1")
    assert_close(w.grad, [2, 4], "w.grad after one backward")
    loss.backward()
    assert_close(w.grad, [4, 8], "w.grad after two backward calls",
                 hint="leaves accumulate (that's why we zero_grad between steps), but intermediate grads must be reset each call")
