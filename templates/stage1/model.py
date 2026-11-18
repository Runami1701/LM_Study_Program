"""Lesson s1-l09 — Assemble the MLP.  ->  python -m course check s1-l09

Build everything from your own layers.py and loss.py functions.
"""

import numpy as np

from .layers import (
    embedding_backward, embedding_forward, flatten_context,
    linear_backward, linear_forward, tanh_backward, tanh_forward,
)
from .loss import cross_entropy, cross_entropy_backward


def init_params(V: int, k: int, e: int, h: int, rng: np.random.Generator) -> dict:
    """Create the parameters, all float32, drawn in exactly this order:

        C  (V, e)    = rng.standard_normal((V, e))
        W1 (k*e, h)  = rng.standard_normal((k*e, h)) / sqrt(k*e)
        W2 (h, V)    = rng.standard_normal((h, V)) * 0.01
        b1 (h,), b2 (V,) = zeros

    Returns:
        dict with keys "C", "W1", "b1", "W2", "b2".
    """
    raise NotImplementedError("init_params")


def predict(params: dict, x: np.ndarray) -> np.ndarray:
    """Logits (B, V) for contexts x (B, k): embed, flatten, linear, tanh, linear."""
    raise NotImplementedError("predict")


def forward(params: dict, x: np.ndarray, y: np.ndarray) -> tuple[float, tuple]:
    """Mean cross-entropy loss of the MLP on (x (B, k), y (B,)).

    Returns:
        loss:  Python float.
        cache: everything backward() needs (your choice).
    """
    raise NotImplementedError("forward")


def backward(cache: tuple) -> dict:
    """Gradients of the loss: dict with the same keys and shapes as params."""
    raise NotImplementedError("backward")


def count_params(params: dict) -> int:
    """Total number of scalars across all parameter arrays (a Python int)."""
    raise NotImplementedError("count_params")
