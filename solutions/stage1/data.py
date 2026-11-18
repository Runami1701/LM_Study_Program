"""Lesson s1-l01 — Vocabulary, encoding and batches (reference solution)."""

import numpy as np


def build_vocab(text: str) -> tuple[dict, list]:
    """Sorted, reproducible character vocabulary: (stoi, itos)."""
    itos = sorted(set(text))
    stoi = {ch: i for i, ch in enumerate(itos)}
    return stoi, itos


def encode(text: str, stoi: dict) -> np.ndarray:
    """String -> (n,) int64 ids."""
    return np.fromiter((stoi[ch] for ch in text), dtype=np.int64, count=len(text))


def decode(ids, itos: list) -> str:
    """Ids -> string."""
    return "".join(itos[int(i)] for i in ids)


def train_val_split(data: np.ndarray, frac: float = 0.9) -> tuple[np.ndarray, np.ndarray]:
    """Position-based split: first int(frac * n) for training."""
    n_train = int(frac * len(data))
    return data[:n_train], data[n_train:]


def get_batch(data: np.ndarray, B: int, k: int, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    """B random windows: x (B, k) int64 contexts, y (B,) int64 next ids."""
    n = len(data)
    starts = rng.integers(0, n - k, size=B)    # (B,)  largest start: n-k-1
    offsets = np.arange(k)                      # (k,)
    idx = starts[:, None] + offsets[None, :]    # (B, 1) + (1, k) -> (B, k)
    x = data[idx].astype(np.int64)              # gather all windows at once
    y = data[starts + k].astype(np.int64)       # (B,)  the next character
    return x, y
