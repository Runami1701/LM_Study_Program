"""Checks for s1-l13 — Sampling."""

import numpy as np
import pytest

from tests.helpers import load

MOD = "stage1.sample"


@pytest.mark.fn("sample_next")
def test_frequencies_match_softmax():
    m = load(MOD)
    logits = np.array([2.0, 1.0, 0.1, -1.0], dtype=np.float32)
    rng = np.random.default_rng(0)
    draws = np.array([m.sample_next(logits, 1.0, rng) for _ in range(20_000)])
    assert all(isinstance(int(d), int) for d in draws[:3])
    freq = np.bincount(draws, minlength=4) / len(draws)
    p = np.exp(logits - logits.max()); p /= p.sum()
    assert np.abs(freq - p).max() < 0.015, f"empirical frequencies {np.round(freq, 3)} should match softmax {np.round(p, 3)}"


@pytest.mark.fn("sample_next")
def test_temperature_limits():
    m = load(MOD)
    logits = np.array([0.5, 2.0, 1.9, -3.0], dtype=np.float32)
    rng = np.random.default_rng(1)
    cold = {m.sample_next(logits, 1e-3, rng) for _ in range(200)}
    assert cold == {1}, f"temperature 1e-3 should always pick the argmax (1), got {sorted(cold)} — any nan?"
    hot = np.bincount([m.sample_next(logits, 1e4, rng) for _ in range(8000)], minlength=4) / 8000
    assert np.abs(hot - 0.25).max() < 0.03, f"temperature 1e4 should be nearly uniform, got {np.round(hot, 3)}"
    out = m.sample_next(logits, 1.0, rng)
    assert isinstance(out, int), f"sample_next should return a Python int, got {type(out).__name__}"


@pytest.mark.fn("sample")
def test_sample_length_vocab_and_determinism():
    m = load(MOD)
    model = load("stage1.model")
    itos = list("\nabcdefgh")
    stoi = {c: i for i, c in enumerate(itos)}
    p = model.init_params(len(itos), 4, 3, 8, np.random.default_rng(0))
    s1 = m.sample(p, itos, stoi, "ab", 50, 1.0, np.random.default_rng(7))
    s2 = m.sample(p, itos, stoi, "ab", 50, 1.0, np.random.default_rng(7))
    assert isinstance(s1, str) and len(s1) == 50, f"expected 50 generated characters, got {len(s1) if isinstance(s1, str) else s1!r}"
    assert set(s1) <= set(itos), "every generated character must come from the vocabulary"
    assert s1 == s2, "the same rng seed must give the same text"


@pytest.mark.fn("sample")
def test_greedy_sliding_window():
    m = load(MOD)
    model = load("stage1.model")
    itos = list("\nxyz")
    stoi = {c: i for i, c in enumerate(itos)}
    p = model.init_params(4, 3, 2, 6, np.random.default_rng(3))
    p["W2"] *= 300  # make the model opinionated so greedy decoding is well defined
    got = m.sample(p, itos, stoi, "x", 12, 1e-4, np.random.default_rng(0))
    ctx = [0, 0, stoi["x"]]
    expected = []
    for _ in range(12):
        nxt = int(np.argmax(model.predict(p, np.array([ctx]))[0]))
        expected.append(itos[nxt])
        ctx = ctx[1:] + [nxt]
    assert got == "".join(expected), (
        f"with a near-zero temperature, expected greedy text {''.join(expected)!r}, got {got!r} — "
        "pad short prompts on the left with the id of '\\n' and slide the window after each draw"
    )
