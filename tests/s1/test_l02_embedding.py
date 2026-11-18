"""Checks for s1-l02 — Embedding lookup (forward)."""

import numpy as np
import pytest

from tests.helpers import assert_close, check_array, load

MOD = "stage1.layers"


def one_hot(x, V):
    """(B, k) ids -> (B, k, V) one-hot, float64 (test-side reference)."""
    return (x[..., None] == np.arange(V)).astype(np.float64)


@pytest.mark.fn("embedding_forward")
def test_embedding_worked_example():
    """The worked example from the lesson: V = 4, e = 2, context 'h e l' = [1, 0, 2]."""
    m = load(MOD)
    C = np.array([[0.1, 0.2], [0.3, -0.4], [0.5, 0.6], [-0.7, 0.8]], dtype=np.float32)
    x = np.array([[1, 0, 2]])
    emb = m.embedding_forward(C, x)
    check_array(emb, "emb", shape=(1, 3, 2), dtype=np.float32, shape_desc="(B, k, e)")
    expected = np.array([[[0.3, -0.4], [0.1, 0.2], [0.5, 0.6]]])
    assert_close(emb, expected, "emb", hint="emb[b, j] should be row x[b, j] of C")


@pytest.mark.fn("embedding_forward")
def test_embedding_equals_one_hot_matmul():
    m = load(MOD)
    rng = np.random.default_rng(0)
    V, e, B, k = 7, 3, 4, 5
    C = rng.standard_normal((V, e)).astype(np.float32)
    x = rng.integers(0, V, size=(B, k))
    x[0, :] = 2  # repeated ids are fine for a lookup
    emb = m.embedding_forward(C, x)
    check_array(emb, "emb", shape=(B, k, e), dtype=np.float32, shape_desc="(B, k, e)")
    expected = one_hot(x, V) @ C.astype(np.float64)  # (B, k, V) @ (V, e) -> (B, k, e)
    assert_close(emb, expected, "emb", hint="the lookup must equal one_hot(x) @ C")


@pytest.mark.fn("embedding_forward")
def test_embedding_accepts_small_int_ids_and_keeps_table():
    m = load(MOD)
    C = np.arange(12, dtype=np.float32).reshape(6, 2)
    C_before = C.copy()
    x = np.array([[5, 0], [3, 3]], dtype=np.uint8)
    emb = m.embedding_forward(C, x)
    check_array(emb, "emb (uint8 ids)", shape=(2, 2, 2), dtype=np.float32)
    assert_close(emb, [[[10, 11], [0, 1]], [[6, 7], [6, 7]]], "emb (uint8 ids)")
    assert np.array_equal(C, C_before), "embedding_forward must not modify the table C"


@pytest.mark.fn("flatten_context")
def test_flatten_concatenates_in_order():
    m = load(MOD)
    emb = np.array(
        [[[1, 2], [3, 4], [5, 6]],
         [[7, 8], [9, 10], [11, 12]]],
        dtype=np.float32,
    )  # (B=2, k=3, e=2)
    flat = m.flatten_context(emb)
    check_array(flat, "flat", shape=(2, 6), dtype=np.float32, shape_desc="(B, k*e)")
    assert_close(flat, [[1, 2, 3, 4, 5, 6], [7, 8, 9, 10, 11, 12]], "flat",
                 hint="row b should be emb[b, 0], then emb[b, 1], ... laid end to end (no transposes)")


@pytest.mark.fn("flatten_context")
def test_flatten_batch_of_one():
    m = load(MOD)
    emb = np.ones((1, 8, 24), dtype=np.float32)
    flat = m.flatten_context(emb)
    check_array(flat, "flat (B = 1)", shape=(1, 192), shape_desc="(B, k*e)",
                hint="keep the batch axis even when B = 1 — reshape to (B, -1), not (-1,)")


@pytest.mark.fn("flatten_context")
def test_flatten_general_shapes():
    m = load(MOD)
    rng = np.random.default_rng(1)
    emb = rng.standard_normal((5, 4, 3)).astype(np.float32)
    flat = m.flatten_context(emb)
    check_array(flat, "flat", shape=(5, 12), dtype=np.float32)
    for b in range(5):
        assert_close(flat[b], np.concatenate(list(emb[b])), f"flat[{b}]")
