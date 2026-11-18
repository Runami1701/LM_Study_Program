"""Checks for s1-l04 — Softmax and cross-entropy (stable)."""

import warnings

import numpy as np
import pytest

from tests.helpers import assert_close, check_array, load

MOD = "stage1.loss"


def ref_softmax(z):
    z = np.asarray(z, dtype=np.float64)
    e = np.exp(z - z.max(axis=-1, keepdims=True))
    return e / e.sum(axis=-1, keepdims=True)


@pytest.mark.fn("softmax")
def test_softmax_worked_example():
    m = load(MOD)
    p = m.softmax(np.array([[2.0, 1.0, 0.1]], dtype=np.float32))
    check_array(p, "softmax([[2, 1, 0.1]])", shape=(1, 3), dtype=np.float32)
    assert_close(p, [[0.659, 0.242, 0.099]], "softmax([[2, 1, 0.1]])", atol=1e-3)


@pytest.mark.fn("softmax")
def test_softmax_rows_sum_to_one_and_match_naive():
    m = load(MOD)
    z = np.random.default_rng(0).standard_normal((6, 10)).astype(np.float32) * 3
    p = m.softmax(z)
    check_array(p, "probs", shape=(6, 10), dtype=np.float32)
    assert_close(p.sum(axis=1), np.ones(6), "row sums", atol=1e-5,
                 hint="normalize over the last axis (keepdims=True keeps the division row-wise)")
    assert_close(p, ref_softmax(z), "probs", atol=1e-6)


@pytest.mark.fn("softmax")
def test_softmax_huge_logits_stable():
    m = load(MOD)
    z = np.array([[1e4, 0.0, -1e4], [1000.0, 1000.0, 999.0]], dtype=np.float32)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        p = m.softmax(z)
    assert np.all(np.isfinite(p)), (
        f"softmax gave {p} for logits like 1e4 — exp(1e4) overflows; subtract each row's max first"
    )
    assert_close(p, ref_softmax(z), "softmax(huge logits)", atol=1e-6)


@pytest.mark.fn("softmax")
def test_softmax_works_on_1d():
    m = load(MOD)
    p = m.softmax(np.array([0.0, np.log(3.0)], dtype=np.float32))
    assert_close(p, [0.25, 0.75], "softmax([0, ln 3])", atol=1e-6,
                 hint="use axis=-1 so a single (V,) vector works too")


@pytest.mark.fn("cross_entropy")
def test_cross_entropy_worked_example():
    m = load(MOD)
    loss, _ = m.cross_entropy(np.array([[2.0, 1.0, 0.1]], dtype=np.float32), np.array([0]))
    assert isinstance(loss, float), f"loss should be a Python float, got {type(loss).__name__}"
    assert abs(loss - 0.417) < 1e-3, f"logits [2, 1, 0.1] with target 0: expected -ln 0.659 = 0.417, got {loss:.4f}"


@pytest.mark.fn("cross_entropy")
def test_cross_entropy_batch_mean():
    m = load(MOD)
    rng = np.random.default_rng(1)
    z = rng.standard_normal((8, 5)).astype(np.float32)
    y = rng.integers(0, 5, 8)
    loss, _ = m.cross_entropy(z, y)
    expected = -np.mean(np.log(ref_softmax(z)[np.arange(8), y]))
    assert abs(loss - expected) < 1e-5, (
        f"expected the batch MEAN of -log p[b, y_b] = {expected:.5f}, got {loss:.5f}"
        + (" (that's the sum — divide by B)" if abs(loss - 8 * expected) < 1e-4 else "")
    )


@pytest.mark.fn("cross_entropy")
def test_cross_entropy_uniform_is_ln_V():
    m = load(MOD)
    for V in (4, 65):
        loss, _ = m.cross_entropy(np.zeros((3, V), dtype=np.float32), np.array([0, 1, 2]))
        assert abs(loss - np.log(V)) < 1e-5, f"uniform logits over V = {V} must give ln V = {np.log(V):.4f}, got {loss:.4f}"


@pytest.mark.fn("cross_entropy")
def test_cross_entropy_huge_logits():
    m = load(MOD)
    z = np.array([[1e4, 0.0, -1e4], [-1e4, 1e4, 0.0]], dtype=np.float32)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        loss, _ = m.cross_entropy(z, np.array([0, 2]))
    assert np.isfinite(loss), (
        f"loss is {loss} for logits around 1e4 — compute log p with log-sum-exp, not log(softmax(...))"
    )
    assert abs(loss - 5000.0) < 1e-2, f"expected (0 + 1e4) / 2 = 5000, got {loss}"
