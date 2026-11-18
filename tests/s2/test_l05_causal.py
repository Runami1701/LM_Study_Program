"""Checks for s2-l05 — Causal mask."""

import numpy as np
import pytest

from tests.a.graph import check_op
from tests.helpers import assert_close, check_array, load
from tests.s2.common import T

MOD = "stage2.attention"


@pytest.mark.fn("causal_mask")
def test_mask_values():
    m = load(MOD)
    M = m.causal_mask(4)
    check_array(M, "causal_mask(4)", shape=(4, 4), dtype=np.float32)
    assert np.all(M[np.tril_indices(4)] == 0), "on and below the diagonal (key <= query) the mask must be 0"
    above = M[np.triu_indices(4, 1)]
    assert np.all(above <= -1e8) and np.all(np.isfinite(above)), (
        "above the diagonal use a large FINITE negative number (e.g. -1e9), not -inf "
        "(-inf - (-inf) = nan in the max-subtraction of a fully masked row)"
    )


@pytest.mark.fn("causal_mask")
def test_future_does_not_leak():
    m = load(MOD)
    rng = np.random.default_rng(0)
    Ws = [T()(rng.standard_normal((6, 3)).astype(np.float32)) for _ in range(3)]
    x = rng.standard_normal((1, 7, 6)).astype(np.float32)
    out1 = m.attention_single_head(T()(x), *Ws, causal=True).data
    for t in range(6):
        x2 = x.copy()
        x2[:, t + 1:] = rng.standard_normal(x2[:, t + 1:].shape)
        out2 = m.attention_single_head(T()(x2), *Ws, causal=True).data
        assert_close(out2[:, :t + 1], out1[:, :t + 1], f"outputs at positions 0..{t} after changing positions > {t}",
                     atol=1e-6, hint="position t must only attend to positions <= t: add causal_mask(T) to the scores")
    out_nc = m.attention_single_head(T()(x), *Ws, causal=False).data
    assert not np.allclose(out_nc, out1), "causal=False must not apply the mask"


@pytest.mark.fn("causal_mask")
def test_first_position_attends_only_to_itself():
    m = load(MOD)
    rng = np.random.default_rng(1)
    x = rng.standard_normal((2, 5, 4)).astype(np.float32)
    I = T()(np.eye(4, dtype=np.float32))
    out = m.attention_single_head(T()(x), I, I, I, causal=True).data
    assert_close(out[:, 0], x[:, 0], "output at position 0 (with identity projections)",
                 atol=1e-6, hint="row 0 of P must be [1, 0, 0, ...]: everything after position 0 is masked")


@pytest.mark.fn("causal_mask")
def test_causal_gradcheck():
    m = load(MOD)
    rng = np.random.default_rng(2)
    check_op(T(), lambda x, a, b, c: m.attention_single_head(x, a, b, c, causal=True),
             [rng.standard_normal((2, 4, 3)), rng.standard_normal((3, 2)), rng.standard_normal((3, 2)), rng.standard_normal((3, 2))],
             rng, "causal attention", tol=1e-5)
