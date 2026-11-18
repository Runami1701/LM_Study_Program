"""Lesson s3-l04 — Bits per character.  ->  python -m course check s3-l04"""

import gc
import math

import numpy as np

from .bpe import build_vocab, encode


def bits_per_char(total_nats: float, n_chars: int) -> float:
    """Turn a summed loss (in nats) over n_chars characters into bits per character."""
    raise NotImplementedError("bits_per_char")


def evaluate_bpc(model, text: str, merges: list, T: int, batch_size: int = 8) -> float:
    """Bits per character of `model` on `text` (characters counted as UTF-8 bytes).

    1. ids = encode(text, merges); n_bytes[i] = number of bytes of token i (build_vocab).
    2. Cut ids into consecutive windows of T + 1 tokens: window r is ids[r*T : r*T + T + 1].
       Inputs x = its first T tokens, targets y = its last T tokens.
    3. For batches of windows: _, loss = model(x, y). loss.data is the MEAN over y.size
       predictions, so the window batch contributes float(loss.data) * y.size nats,
       and n_bytes[y].sum() characters. Call gc.collect() after each batch.
    4. Return bits_per_char(total nats, total characters).
    """
    raise NotImplementedError("evaluate_bpc")
