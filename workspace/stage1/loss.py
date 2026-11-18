"""Softmax and cross-entropy.

Lesson s1-l04: softmax, cross_entropy     ->  python -m course check s1-l04
Lesson s1-l05: cross_entropy_backward     ->  python -m course check s1-l05
"""

import numpy as np


def softmax(logits: np.ndarray) -> np.ndarray:
    """Softmax over the last axis.

    Args:
        logits: (..., V) float32; may contain huge values such as 1e4.

    Returns:
        probs, same shape and dtype; every row is positive and sums to 1.
        Must not produce nan or inf for any finite input.
    """
    raise NotImplementedError("softmax")


def cross_entropy(logits: np.ndarray, y: np.ndarray) -> tuple[float, tuple]:
    """Mean cross-entropy loss of a batch.

    Args:
        logits: (B, V) float32 scores.
        y:      (B,) int64 target ids.

    Returns:
        loss:  Python float, L = -(1/B) * sum_b log p[b, y[b]], with p = softmax(logits),
               computed stably (log-sum-exp): no nan/inf for logits around 1e4.
        cache: whatever cross_entropy_backward needs (your choice, e.g. (probs, y)).
    """
    raise NotImplementedError("cross_entropy")


def cross_entropy_backward(cache: tuple) -> np.ndarray:
    """Gradient of the mean cross-entropy with respect to the logits.

    Returns:
        dlogits: (B, V), same dtype as the logits.
    """
    raise NotImplementedError("cross_entropy_backward")
