"""Checks for s2-l03 — GELU."""

import numpy as np
import pytest

from tests.a.graph import check_op
from tests.helpers import assert_close, check_array, load
from tests.s2.common import T, ref_gelu


@pytest.mark.fn("gelu")
def test_values_and_dtype():
    m = load("stage2.layers")
    x = np.array([-3.0, -1.0, 0.0, 0.5, 1.0, 3.0], dtype=np.float32)
    out = m.gelu(T()(x))
    check_array(out.data, "gelu(x)", shape=(6,), dtype=np.float32,
                hint="a NumPy float64 constant (np.sqrt(2 / np.pi)) upcasts everything — use math.sqrt")
    assert_close(out.data, ref_gelu(x.astype(np.float64)), "gelu(x)", atol=1e-6)
    assert_close(out.data[[2, 5]], [0.0, 2.99636], "gelu(0), gelu(3)", atol=1e-4)


@pytest.mark.fn("gelu")
def test_gradcheck():
    m = load("stage2.layers")
    rng = np.random.default_rng(0)
    check_op(T(), m.gelu, [rng.standard_normal((3, 7)) * 2], rng, "gelu", tol=1e-5)
