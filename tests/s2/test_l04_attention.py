"""Checks for s2-l04 — Single-head attention."""

import numpy as np
import pytest

from tests.a.graph import check_op
from tests.helpers import assert_close, check_array, load
from tests.s2.common import T, ref_softmax

MOD = "stage2.attention"


def ref_attention(x, Wq, Wk, Wv, causal=False, scale=True):
    q, k, v = x @ Wq, x @ Wk, x @ Wv
    s = q @ np.swapaxes(k, -1, -2)
    if scale:
        s = s / np.sqrt(q.shape[-1])
    if causal:
        Tn = x.shape[1]
        s = s + np.triu(np.full((Tn, Tn), -1e9), 1)
    return ref_softmax(s) @ v


@pytest.mark.fn("softmax")
def test_softmax_op():
    m = load(MOD)
    rng = np.random.default_rng(0)
    z = rng.standard_normal((2, 3, 5)).astype(np.float32)
    out = m.softmax(T()(z))
    check_array(out.data, "softmax", shape=(2, 3, 5), dtype=np.float32)
    assert_close(out.data, ref_softmax(z.astype(np.float64)), "softmax", atol=1e-6)
    huge = m.softmax(T()(np.array([[1e4, 0.0, -1e4]], dtype=np.float32)))
    assert np.all(np.isfinite(huge.data)), "softmax must subtract the row max (stable for logits ~1e4)"
    check_op(T(), m.softmax, [rng.standard_normal((2, 4, 3))], rng, "softmax")


@pytest.mark.fn("attention_single_head")
def test_attention_values():
    m = load(MOD)
    rng = np.random.default_rng(1)
    x = rng.standard_normal((2, 5, 8)).astype(np.float32)
    Ws = [rng.standard_normal((8, 4)).astype(np.float32) * 0.5 for _ in range(3)]
    out = m.attention_single_head(T()(x), *[T()(w) for w in Ws])
    check_array(out.data, "attention output", shape=(2, 5, 4), dtype=np.float32, shape_desc="(B, T, d_h)")
    ref = ref_attention(*(a.astype(np.float64) for a in [x] + Ws))
    if not np.allclose(out.data, ref, atol=1e-5):
        unscaled = ref_attention(*(a.astype(np.float64) for a in [x] + Ws), scale=False)
        hint = " — it matches attention WITHOUT the 1/sqrt(d_h) scale" if np.allclose(out.data, unscaled, atol=1e-5) else ""
        assert_close(out.data, ref, "attention output", atol=1e-5, hint=("softmax(q k^T / sqrt(d_h)) v" + hint))


@pytest.mark.fn("attention_single_head")
def test_attention_gradcheck():
    m = load(MOD)
    rng = np.random.default_rng(2)
    check_op(T(), lambda x, a, b, c: m.attention_single_head(x, a, b, c),
             [rng.standard_normal((2, 3, 4)), rng.standard_normal((4, 2)), rng.standard_normal((4, 2)), rng.standard_normal((4, 2))],
             rng, "attention_single_head", tol=1e-5)


@pytest.mark.fn("attention_single_head")
def test_scale_keeps_softmax_soft_at_large_dh():
    """Unit-variance q, k with d_h = 256: q.k has std 16 unscaled (saturated softmax), 1 scaled."""
    m = load(MOD)
    rng = np.random.default_rng(3)
    d = 256
    x = rng.standard_normal((1, 16, d)).astype(np.float32)
    I = np.eye(d, dtype=np.float32)
    out = m.attention_single_head(T()(x), T()(I), T()(I), T()(I))
    ref = ref_attention(x.astype(np.float64), I, I, I)
    assert_close(out.data, ref, "attention output at d_h = 256", atol=1e-4,
                 hint="divide the scores by sqrt(d_h) before the softmax")
