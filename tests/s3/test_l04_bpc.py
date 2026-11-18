"""Checks for s3-l04 — Bits per character."""

import math

import numpy as np
import pytest

from tests.helpers import load

MOD = "stage3.metrics"


class Uniform:
    """A fake model whose loss is always ln V (it guesses uniformly over V tokens)."""

    def __init__(self, V):
        self.V = V
        self.calls = 0

    def __call__(self, x, y):
        self.calls += 1

        class L:
            data = np.float32(math.log(self.V))

        return None, L()


@pytest.mark.fn("bits_per_char")
def test_conversion():
    m = load(MOD)
    assert abs(m.bits_per_char(100 * math.log(2), 100) - 1.0) < 1e-12, "100·ln 2 nats over 100 characters is exactly 1 bit per character"
    assert abs(m.bits_per_char(3.0, 2) - 3.0 / (2 * math.log(2))) < 1e-12


@pytest.mark.fn("evaluate_bpc")
def test_uniform_byte_model_scores_8_bits():
    m = load(MOD)
    text = "Once upon a time there was a cat. " * 20
    bpc = m.evaluate_bpc(Uniform(256), text, [], T=16)
    assert abs(bpc - 8.0) < 1e-4, f"a uniform guess over 256 byte tokens is log2(256) = 8 bits per character, got {bpc:.4f}"


@pytest.mark.fn("evaluate_bpc")
def test_counts_bytes_of_merged_tokens():
    m = load(MOD)
    from solutions.stage3.bpe import encode, train_bpe
    text = "the cat sat on the mat. " * 40
    merges = train_bpe(text, 300)
    n_tokens = len(encode(text, merges))
    model = Uniform(300)
    bpc = m.evaluate_bpc(model, text, merges, T=8)
    windows = (n_tokens - 1) // 8
    assert model.calls == math.ceil(windows / 8), "use windows of T + 1 tokens, in batches of batch_size windows"
    assert bpc < math.log2(300), (
        "with merged tokens, each prediction covers several characters, so bits per character must be below log2(V)"
    )
