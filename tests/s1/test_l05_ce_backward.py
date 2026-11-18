"""Checks for s1-l05 — Cross-entropy gradient."""

import numpy as np
import pytest

from tests.helpers import assert_close, check_array, load, numerical_grad, rel_error

MOD = "stage1.loss"


def ce64(z, y):
    z = z - z.max(axis=1, keepdims=True)
    lse = np.log(np.exp(z).sum(axis=1))
    return float(np.mean(lse - z[np.arange(len(y)), y]))


@pytest.mark.fn("cross_entropy_backward")
def test_worked_example():
    m = load(MOD)
    _, cache = m.cross_entropy(np.array([[2.0, 1.0, 0.1]], dtype=np.float32), np.array([0]))
    d = m.cross_entropy_backward(cache)
    check_array(d, "dlogits", shape=(1, 3), dtype=np.float32)
    assert_close(d, [[0.659 - 1, 0.242, 0.099]], "dlogits", atol=1e-3,
                 hint="dL/dz_i = p_i - 1[i = y]")


@pytest.mark.fn("cross_entropy_backward")
def test_matches_numerical_gradient():
    m = load(MOD)
    rng = np.random.default_rng(0)
    z = rng.standard_normal((4, 6))
    y = rng.integers(0, 6, 4)
    _, cache = m.cross_entropy(z.copy(), y)
    d = m.cross_entropy_backward(cache)
    num = numerical_grad(lambda: ce64(z, y), z)
    err = rel_error(d, num)
    assert err < 1e-6, f"dlogits differs from the numerical gradient (max relative error {err:.2e})" + (
        " — looks like the 1/B factor is missing" if np.allclose(d, 4 * num, atol=1e-6) else "")


@pytest.mark.fn("cross_entropy_backward")
def test_rows_sum_to_zero():
    m = load(MOD)
    z = np.random.default_rng(2).standard_normal((5, 7)).astype(np.float32)
    _, cache = m.cross_entropy(z, np.array([0, 1, 2, 3, 4]))
    d = m.cross_entropy_backward(cache)
    assert_close(d.sum(axis=1), np.zeros(5), "row sums of dlogits", atol=1e-6)


@pytest.mark.fn("cross_entropy_backward")
def test_gradient_independent_of_batch_size_scale():
    m = load(MOD)
    z = np.random.default_rng(3).standard_normal((1, 4)).astype(np.float32)
    _, c1 = m.cross_entropy(z, np.array([2]))
    _, c8 = m.cross_entropy(np.repeat(z, 8, axis=0), np.full(8, 2))
    d1 = m.cross_entropy_backward(c1)
    d8 = m.cross_entropy_backward(c8)
    assert_close(d8.sum(axis=0), d1[0], "sum over 8 identical rows",
                 hint="the loss is a mean, so each row's gradient carries a 1/B factor")
