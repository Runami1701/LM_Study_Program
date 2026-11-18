"""Softmax and cross-entropy (reference solution)."""

import numpy as np


def softmax(logits: np.ndarray) -> np.ndarray:
    """Row-wise softmax over the last axis, stable for huge logits."""
    z = logits - logits.max(axis=-1, keepdims=True)   # shift: largest entry becomes 0
    e = np.exp(z)                                      # every entry in (0, 1]
    return e / e.sum(axis=-1, keepdims=True)


def cross_entropy(logits: np.ndarray, y: np.ndarray) -> tuple[float, tuple]:
    """Mean negative log-probability of the targets; cache = (probs, y)."""
    B = logits.shape[0]
    z = logits - logits.max(axis=1, keepdims=True)     # (B, V)
    lse = np.log(np.exp(z).sum(axis=1))                # (B,)  log-sum-exp of shifted logits
    logp_y = z[np.arange(B), y] - lse                  # (B,)  log p_{b, y_b}
    loss = float(-logp_y.mean())
    probs = np.exp(z - lse[:, None])                   # softmax, reusing lse
    return loss, (probs, y)


def cross_entropy_backward(cache: tuple) -> np.ndarray:
    """dL/dlogits = (probs - one_hot(y)) / B."""
    probs, y = cache
    B = probs.shape[0]
    dlogits = probs.copy()
    dlogits[np.arange(B), y] -= 1.0
    return dlogits / B
