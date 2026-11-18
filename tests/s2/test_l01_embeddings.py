"""Checks for s2-l01 — Token + positional embeddings, next-token targets."""

import numpy as np
import pytest

from tests.a.graph import run_backward
from tests.helpers import assert_close, assert_equal, check_array, load
from tests.s2.common import T

MOD = "stage2.layers"


@pytest.mark.fn("get_batch_seq")
def test_targets_are_shifted_inputs():
    m = load(MOD)
    data = np.arange(500, dtype=np.int64)
    x, y = m.get_batch_seq(data, 6, 10, np.random.default_rng(0))
    check_array(x, "x", shape=(6, 10), dtype=np.int64, shape_desc="(B, T)")
    check_array(y, "y", shape=(6, 10), dtype=np.int64, shape_desc="(B, T)")
    assert_equal(x, x[:, :1] + np.arange(10), "x", hint="each row is T consecutive ids")
    assert_equal(y, x + 1, "y", hint="y[:, t] must be the id right after x[:, t] — the whole window shifted by one")


@pytest.mark.fn("get_batch_seq")
def test_seeded_starts_and_dtype():
    m = load(MOD)
    data = (np.random.default_rng(1).integers(0, 65, 2000)).astype(np.uint16)
    x, y = m.get_batch_seq(data, 8, 16, np.random.default_rng(5))
    check_array(x, "x (from uint16 data)", shape=(8, 16), dtype=np.int64)
    starts = np.random.default_rng(5).integers(0, len(data) - 16, size=8)
    for r, s in enumerate(starts):
        assert np.array_equal(x[r], data[s:s + 16]) and np.array_equal(y[r], data[s + 1:s + 17]), (
            f"row {r}: expected the window starting at {s} (rng.integers(0, n - T, size=B))"
        )


@pytest.mark.fn("Linear")
def test_linear():
    m = load(MOD)
    lin = m.Linear(3, 4, np.random.default_rng(0))
    W = np.random.default_rng(0).standard_normal((3, 4)) * 0.02
    assert_close(lin.W.data, W, "Linear.W", hint="W = rng.standard_normal((n_in, n_out)) * 0.02")
    assert lin.W.data.dtype == np.float32 and lin.b.data.dtype == np.float32, "parameters must be float32"
    assert not lin.b.data.any() and lin.b.data.shape == (4,), "b starts as zeros (n_out,)"
    x = T()(np.random.default_rng(1).standard_normal((2, 5, 3)).astype(np.float32))
    out = lin(x)
    assert_close(out.data, x.data @ lin.W.data, "Linear output on (2, 5, 3) input")
    nob = m.Linear(3, 4, np.random.default_rng(0), bias=False)
    assert nob.b is None, "with bias=False, self.b must be None"
    assert len(nob.named_parameters()) == 1, "a bias-free Linear has exactly one parameter"


@pytest.mark.fn("Embeddings")
def test_embeddings_values_and_shapes():
    m = load(MOD)
    emb = m.Embeddings(11, 8, 4, np.random.default_rng(0))
    rng = np.random.default_rng(0)
    wte = rng.standard_normal((11, 4)) * 0.02
    wpe = rng.standard_normal((8, 4)) * 0.02
    assert_close(emb.wte.data, wte, "wte", hint="draw wte (V, d) first, then wpe (T, d)")
    assert_close(emb.wpe.data, wpe, "wpe")
    idx = np.array([[1, 2, 2, 10, 0], [3, 3, 3, 3, 3]])
    out = emb(idx)
    check_array(out.data, "embeddings output", shape=(2, 5, 4), dtype=np.float32, shape_desc="(B, t, d)")
    assert_close(out.data, wte[idx] + wpe[:5], "embeddings output",
                 hint="token vector of each id plus the position vector of its position (0..t-1)")


@pytest.mark.fn("Embeddings")
def test_embeddings_backward():
    m = load(MOD)
    emb = m.Embeddings(5, 4, 2, np.random.default_rng(0))
    idx = np.array([[1, 1, 3], [0, 1, 4]])
    out = emb(idx)
    G = np.random.default_rng(1).standard_normal((2, 3, 2)).astype(np.float32)
    run_backward(out, G)
    exp_wte = np.zeros((5, 2))
    for b in range(2):
        for t in range(3):
            exp_wte[idx[b, t]] += G[b, t]
    exp_wpe = np.zeros((4, 2))
    exp_wpe[:3] = G.sum(axis=0)
    assert_close(emb.wte.grad, exp_wte, "wte.grad", atol=1e-6, hint="repeated ids must accumulate")
    assert_close(emb.wpe.grad, exp_wpe, "wpe.grad", atol=1e-6,
                 hint="each position vector was broadcast over the batch: its gradient sums over B; unused positions get 0")
