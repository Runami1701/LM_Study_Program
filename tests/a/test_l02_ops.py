"""Checks for a-l02 — Broadcasting-aware ops."""

import numpy as np
import pytest

from tests.a.graph import check_op, run_backward
from tests.helpers import assert_close, load


def T():
    return load("autograd.tensor").Tensor


def ops():
    return load("autograd.ops")


@pytest.mark.fn("unbroadcast")
def test_unbroadcast_cases():
    ub = load("autograd.tensor").unbroadcast
    g = np.arange(6.0).reshape(2, 3)
    assert_close(ub(g, (3,)), [3, 5, 7], "unbroadcast((2,3) -> (3,))",
                 hint="b (3,) was stretched over 2 rows: sum the rows")
    assert_close(ub(g, (2, 1)), [[3], [12]], "unbroadcast((2,3) -> (2,1))",
                 hint="a length-1 axis was stretched: sum it but keep it (keepdims)")
    assert_close(ub(g, (1, 3)), [[3, 5, 7]], "unbroadcast((2,3) -> (1,3))")
    assert_close(ub(g, (2, 3)), g, "unbroadcast to the same shape")
    g4 = np.ones((4, 2, 3))
    assert_close(ub(g4, (1, 3)), [[8, 8, 8]], "unbroadcast((4,2,3) -> (1,3))")
    assert ub(g4, ()).shape == (), "unbroadcast to () (a scalar) should sum everything"


@pytest.mark.fn("unbroadcast")
def test_add_mul_broadcast():
    Tn = T()
    rng = np.random.default_rng(0)
    check_op(Tn, lambda a, b: a + b, [rng.standard_normal((4, 3)), rng.standard_normal(3)], rng, "a + b (bias broadcast)")
    check_op(Tn, lambda a, b: a * b, [rng.standard_normal((2, 1, 3)), rng.standard_normal((4, 1))], rng, "a * b (both broadcast)")


@pytest.mark.fn("matmul")
def test_matmul():
    Tn, o = T(), ops()
    rng = np.random.default_rng(1)
    check_op(Tn, o.matmul, [rng.standard_normal((4, 3)), rng.standard_normal((3, 5))], rng, "matmul 2-D")
    check_op(Tn, o.matmul, [rng.standard_normal((2, 4, 3)), rng.standard_normal((3, 5))], rng, "matmul batched @ 2-D")
    check_op(Tn, o.matmul, [rng.standard_normal((2, 2, 4, 3)), rng.standard_normal((2, 2, 3, 4))], rng, "matmul 4-D batched")


@pytest.mark.fn("tanh")
def test_tanh():
    rng = np.random.default_rng(2)
    check_op(T(), ops().tanh, [rng.standard_normal((3, 4))], rng, "tanh")


@pytest.mark.fn("sum")
def test_sum():
    Tn, o = T(), ops()
    rng = np.random.default_rng(3)
    x = rng.standard_normal((2, 3, 4))
    check_op(Tn, lambda a: o.sum(a), [x], rng, "sum (all)")
    check_op(Tn, lambda a: o.sum(a, axis=1), [x], rng, "sum(axis=1)")
    check_op(Tn, lambda a: o.sum(a, axis=-1, keepdims=True), [x], rng, "sum(axis=-1, keepdims=True)")
    check_op(Tn, lambda a: o.sum(a, axis=(0, 2)), [x], rng, "sum(axis=(0, 2))")


@pytest.mark.fn("mean")
def test_mean():
    Tn, o = T(), ops()
    rng = np.random.default_rng(4)
    x = rng.standard_normal((3, 5))
    out = o.mean(Tn(x), axis=1)
    assert_close(out.data, x.mean(axis=1), "mean(axis=1)")
    check_op(Tn, lambda a: o.mean(a), [x], rng, "mean (all)")
    check_op(Tn, lambda a: o.mean(a, axis=-1, keepdims=True), [x], rng, "mean(axis=-1, keepdims=True)")


@pytest.mark.fn("reshape")
def test_reshape():
    rng = np.random.default_rng(5)
    check_op(T(), lambda a: ops().reshape(a, (3, -1)), [rng.standard_normal((2, 3, 2))], rng, "reshape")


@pytest.mark.fn("transpose")
def test_transpose():
    rng = np.random.default_rng(6)
    check_op(T(), lambda a: ops().transpose(a, (2, 0, 1)), [rng.standard_normal((2, 3, 4))], rng, "transpose (2, 0, 1)",)


@pytest.mark.fn("getitem")
def test_getitem_embedding_with_repeats():
    Tn, o = T(), ops()
    rng = np.random.default_rng(7)
    idx = np.array([[0, 3, 3], [4, 0, 3]])
    check_op(Tn, lambda C: o.getitem(C, idx), [rng.standard_normal((5, 2))], rng, "getitem (embedding lookup)")
    check_op(Tn, lambda a: o.getitem(a, (slice(None), slice(1, 3))), [rng.standard_normal((3, 4))], rng, "getitem (slice)")
    C = Tn(np.zeros((3, 2)))
    out = o.getitem(C, np.array([1, 1]))
    run_backward(out, np.ones((2, 2)))
    assert_close(C.grad[1], [2, 2], "grad of a row looked up twice", hint="repeated indices must accumulate: np.add.at")


@pytest.mark.fn("cross_entropy")
def test_cross_entropy():
    Tn, o = T(), ops()
    rng = np.random.default_rng(8)
    y = np.array([0, 2, 1, 2])
    z = rng.standard_normal((4, 3))
    out = o.cross_entropy(Tn(z), y)
    assert out.data.shape == (), f"cross_entropy should return a scalar Tensor (shape ()), got {out.data.shape}"
    zz = z - z.max(1, keepdims=True)
    ref = np.mean(np.log(np.exp(zz).sum(1)) - zz[np.arange(4), y])
    assert abs(float(out.data) - ref) < 1e-6, f"loss {float(out.data):.6f}, expected {ref:.6f}"
    check_op(Tn, lambda a: o.cross_entropy(a, y), [z], rng, "cross_entropy")
    big = o.cross_entropy(Tn(np.array([[1e4, 0.0]], dtype=np.float32)), np.array([0]))
    assert np.isfinite(big.data), "cross_entropy must be stable for huge logits (subtract the row max)"
