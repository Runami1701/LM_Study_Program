"""Lesson s1-l01 — Vocabulary, encoding and batches.

Implement the functions below, then run:  python -m course check s1-l01
No Python loops over batch rows: build whole batches with array operations.
"""

import numpy as np


def build_vocab(text: str) -> tuple[dict, list]:
    """Build the character vocabulary of `text`.

    Returns:
        stoi: dict mapping each distinct character to an integer id.
        itos: list where itos[i] is the character with id i.
        Ids are assigned in sorted character order: the smallest character
        gets id 0. len(itos) == V, the number of distinct characters.
    """
    raise NotImplementedError("build_vocab")


def encode(text: str, stoi: dict) -> np.ndarray:
    """Encode a string as an array of ids.

    Returns:
        ids: (n,) int64 array, where n = len(text) and ids[t] = stoi[text[t]].
    """
    raise NotImplementedError("encode")


def decode(ids, itos: list) -> str:
    """Turn a sequence of ids (list or 1-D array of ints) back into a string."""
    raise NotImplementedError("decode")


def train_val_split(data: np.ndarray, frac: float = 0.9) -> tuple[np.ndarray, np.ndarray]:
    """Split by position: the first int(frac * n) ids train, the rest validate.

    Returns:
        (train, val), in their original order, with len(train) + len(val) == n.
    """
    raise NotImplementedError("train_val_split")


def get_batch(data: np.ndarray, B: int, k: int, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    """Sample B random (context, next-character) examples from `data`.

    Args:
        data: (n,) array of ids (any integer dtype, e.g. uint8 or int64).
        B: batch size.
        k: context length.
        rng: NumPy Generator. Draw the starting positions with exactly one call:
             starts = rng.integers(0, n - k, size=B)
             (the checks reproduce your batch with the same call.)

    Returns:
        x: (B, k) int64, x[r] = data[starts[r] : starts[r] + k]
        y: (B,)   int64, y[r] = data[starts[r] + k]
    """
    raise NotImplementedError("get_batch")
