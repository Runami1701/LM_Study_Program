"""Checks for s2-l02 — LayerNorm."""

import numpy as np
import pytest

from tests.a.graph import check_op
from tests.helpers import assert_close, check_array, load
from tests.s2.common import T, ref_layernorm

MOD = "stage2.layers"


@pytest.mark.fn("layernorm")
def test_normalizes_rows():
    m = load(MOD)
    Tn = T()
    x = np.random.default_rng(0).standard_normal((4, 6, 16)).astype(np.float32) * 5 + 3
    out = m.layernorm(Tn(x), Tn(np.ones(16, np.float32)), Tn(np.zeros(16, np.float32)))
    check_array(out.data, "layernorm output", shape=(4, 6, 16), dtype=np.float32)
    assert_close(out.data.mean(-1), np.zeros((4, 6)), "row means", atol=1e-5)
    assert_close(out.data.var(-1), np.ones((4, 6)), "row variances", atol=1e-3,
                 hint="normalize over the LAST axis, with the biased variance (mean of squared deviations)")


@pytest.mark.fn("layernorm")
def test_matches_formula_with_gamma_beta():
    m = load(MOD)
    Tn = T()
    rng = np.random.default_rng(1)
    x, g, b = rng.standard_normal((3, 5)), rng.standard_normal(5), rng.standard_normal(5)
    out = m.layernorm(Tn(x), Tn(g), Tn(b))
    assert_close(out.data, ref_layernorm(x, g, b), "layernorm(x, gamma, beta)", atol=1e-6)


@pytest.mark.fn("layernorm")
def test_gradcheck():
    m = load(MOD)
    rng = np.random.default_rng(2)
    check_op(T(), m.layernorm, [rng.standard_normal((2, 3, 5)), rng.standard_normal(5), rng.standard_normal(5)],
             rng, "layernorm", tol=1e-5)


@pytest.mark.fn("LayerNorm")
def test_module():
    m = load(MOD)
    ln = m.LayerNorm(8)
    assert np.array_equal(ln.gamma.data, np.ones(8)) and not ln.beta.data.any(), "gamma starts at 1, beta at 0"
    assert ln.gamma.data.dtype == np.float32, "parameters are float32"
    assert set(ln.named_parameters()) == {"gamma", "beta"}
    x = T()(np.random.default_rng(3).standard_normal((2, 8)).astype(np.float32))
    assert_close(ln(x).data, ref_layernorm(x.data.astype(np.float64), 1, 0), "LayerNorm(x)", atol=1e-5)
