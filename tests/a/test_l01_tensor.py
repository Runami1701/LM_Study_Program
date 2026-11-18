"""Checks for a-l01 — The Tensor class."""

import numpy as np
import pytest

from tests.helpers import assert_close, check_array, load

MOD = "autograd.tensor"


@pytest.mark.fn("Tensor")
def test_construction():
    T = load(MOD).Tensor
    t = T([1, 2, 3])
    check_array(t.data, "Tensor([1, 2, 3]).data", shape=(3,), dtype=np.float32,
                hint="convert ints and lists to float32")
    check_array(t.grad, "grad", shape=(3,), dtype=np.float32)
    assert not t.grad.any(), "grad starts at zero"
    assert t._prev == () or len(t._prev) == 0, "a fresh tensor has no parents"
    d = T(np.ones((2, 2)))
    assert d.data.dtype == np.float64, "a float64 array stays float64 (gradient checks need it)"
    t._backward()  # a leaf's _backward does nothing, but must exist


@pytest.mark.fn("Tensor")
def test_add_forward_backward():
    T = load(MOD).Tensor
    a, b = T(np.array([1.0, 2.0])), T(np.array([10.0, 20.0]))
    c = a + b
    assert isinstance(c, T), "a + b must return a Tensor"
    assert_close(c.data, [11, 22], "(a + b).data")
    assert set(map(id, c._prev)) == {id(a), id(b)}, "the output's _prev must be (a, b)"
    c.grad = np.array([1.0, -3.0])
    c._backward()
    assert_close(a.grad, [1, -3], "a.grad", hint="d(a+b)/da = 1, so a receives the upstream gradient")
    assert_close(b.grad, [1, -3], "b.grad")


@pytest.mark.fn("Tensor")
def test_mul_forward_backward_and_scalars():
    T = load(MOD).Tensor
    a, b = T(np.array([2.0, 3.0])), T(np.array([5.0, -1.0]))
    c = a * b
    assert_close(c.data, [10, -3], "(a * b).data")
    c.grad = np.array([1.0, 2.0])
    c._backward()
    assert_close(a.grad, [5, -2], "a.grad", hint="d(ab)/da = b, times the upstream gradient")
    assert_close(b.grad, [2, 6], "b.grad", hint="d(ab)/db = a")
    d = a * 3.0
    assert_close(d.data, [6, 9], "a * 3.0", hint="wrap plain numbers with self._wrap")


@pytest.mark.fn("Tensor")
def test_reused_node_accumulates():
    T = load(MOD).Tensor
    x = T(np.array([3.0]))
    f = x * x
    f.grad = np.array([1.0])
    f._backward()
    assert_close(x.grad, [6.0], "x.grad for f = x * x",
                 hint="x is BOTH parents: two contributions of 3 must add up to 2x = 6 — use +=, not =")


@pytest.mark.fn("Tensor")
def test_zero_grad():
    T = load(MOD).Tensor
    x = T(np.array([1.0, 2.0], dtype=np.float32))
    x.grad += 5
    x.zero_grad()
    check_array(x.grad, "grad after zero_grad", shape=(2,), dtype=np.float32)
    assert not x.grad.any(), "zero_grad must reset grad to zeros"
