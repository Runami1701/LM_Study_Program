"""Bits per character (reference solution)."""

import gc
import math

import numpy as np

from .bpe import build_vocab, encode


def bits_per_char(total_nats: float, n_chars: int) -> float:
    """Convert a summed loss in nats into bits per character."""
    return total_nats / (math.log(2) * n_chars)


def evaluate_bpc(model, text: str, merges: list, T: int, batch_size: int = 8) -> float:
    """Bits per character (UTF-8 byte) of `model` on `text`.

    Cut the token stream into consecutive windows of T + 1 tokens. In each window
    the model predicts tokens 1..T; add up their losses (in nats) and the number of
    bytes those predicted tokens stand for.
    """
    ids = np.asarray(encode(text, merges), dtype=np.int64)
    n_bytes = np.array([len(b) for b in build_vocab(merges)])
    n_windows = (len(ids) - 1) // T
    total_nats, total_chars = 0.0, 0
    for start in range(0, n_windows, batch_size):
        rows = range(start, min(start + batch_size, n_windows))
        x = np.stack([ids[r * T: r * T + T] for r in rows])
        y = np.stack([ids[r * T + 1: r * T + T + 1] for r in rows])
        _, loss = model(x, y)
        total_nats += float(loss.data) * y.size        # the loss is a mean over y.size predictions
        total_chars += int(n_bytes[y].sum())
        del loss
        gc.collect()
    return bits_per_char(total_nats, total_chars)
