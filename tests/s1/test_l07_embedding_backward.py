"""Checks for s1-l07 — Embedding backward."""

import numpy as np
import pytest

from tests.helpers import assert_close, check_array, load, numerical_grad, rel_error

MOD = "stage1.layers"


@pytest.mark.fn("embedding_backward")
def test_repeated_ids_accumulate():
    m = load(MOD)
    demb = np.array([[[1.0, 2.0], [10.0, 20.0]]], dtype=np.float32)  # (B=1, k=2, e=2)
    x = np.array([[1, 1]])
    dC = m.embedding_backward(demb, x, 3)
    check_array(dC, "dC", shape=(3, 2), dtype=np.float32, shape_desc="(V, e)")
    assert_close(dC, [[0, 0], [11, 22], [0, 0]], "dC",
                 hint="id 1 was used twice, so its row must receive BOTH gradients; "
                      "dC[x] += demb keeps only one of them — use np.add.at")


@pytest.mark.fn("embedding_backward")
def test_matches_loop_reference():
    m = load(MOD)
    rng = np.random.default_rng(0)
    V, B, k, e = 6, 5, 4, 3
    x = rng.integers(0, V, (B, k))
    demb = rng.standard_normal((B, k, e)).astype(np.float32)
    expected = np.zeros((V, e))
    for b in range(B):
        for j in range(k):
            expected[x[b, j]] += demb[b, j]
    dC = m.embedding_backward(demb, x, V)
    check_array(dC, "dC", shape=(V, e), dtype=np.float32)
    assert_close(dC, expected, "dC", atol=1e-5)


@pytest.mark.fn("embedding_backward")
def test_gradcheck():
    m = load(MOD)
    rng = np.random.default_rng(1)
    C = rng.standard_normal((5, 2))
    x = np.array([[0, 3, 3], [4, 0, 3]])
    G = rng.standard_normal((2, 3, 2))
    dC = m.embedding_backward(G, x, 5)
    err = rel_error(dC, numerical_grad(lambda: float(np.sum(G * C[x])), C))
    assert err < 1e-7, f"dC differs from the numerical gradient (max relative error {err:.2e})"


@pytest.mark.fn("embedding_backward")
def test_unused_rows_are_zero():
    m = load(MOD)
    dC = m.embedding_backward(np.ones((2, 2, 4), np.float32), np.array([[0, 0], [0, 0]]), 7)
    assert_close(dC[1:], np.zeros((6, 4)), "dC rows of unused ids")
    assert_close(dC[0], np.full(4, 4.0), "dC[0] (id 0 used 4 times)")
