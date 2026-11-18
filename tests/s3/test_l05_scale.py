"""Checks for s3-l05 — Compute budget."""

import pytest

from tests.helpers import load

MOD = "stage3.train_bpe_lm"


@pytest.mark.fn("estimate_hours")
def test_estimate_hours():
    m = load(MOD)
    h = m.estimate_hours(7_000_000, 20_000_000, 50e9)
    assert abs(h - 6 * 7e6 * 2e7 / 50e9 / 3600) < 1e-9, f"6·N·D / FLOPS / 3600 = 4.667 hours, got {h}"


@pytest.mark.fn("tokens_for_hours")
def test_tokens_for_hours_inverts():
    m = load(MOD)
    D = m.tokens_for_hours(7_000_000, 8.0, 20e9)
    assert isinstance(D, int), "return a whole number of tokens"
    assert abs(D - 8 * 3600 * 20e9 / (6 * 7e6)) <= 1, f"expected about 13.7M tokens, got {D:,}"
    assert abs(m.estimate_hours(7_000_000, D, 20e9) - 8.0) < 1e-3, "tokens_for_hours and estimate_hours must undo each other"
