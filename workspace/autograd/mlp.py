"""Lesson a-l04 — the stage-1 MLP rebuilt on the engine.  ->  python -m course check a-l04

Write only the forward pass: the engine does the backward pass for you.
"""

import numpy as np

from . import ops
from .tensor import Tensor


class MLP:
    """Same model as stage 1, built from a params dict (e.g. stage-1 init_params).

    Usage:
        model = MLP(params)
        loss = model(x, y)          # scalar Tensor
        loss.backward()             # fills model.parameters()[name].grad
    """

    def __init__(self, params: dict):
        """Wrap each array of params ("C", "W1", "b1", "W2", "b2") in a Tensor."""
        raise NotImplementedError("MLP")

    def parameters(self) -> dict:
        """Dict name -> Tensor, same names as the stage-1 params."""
        raise NotImplementedError("MLP")

    def __call__(self, x: np.ndarray, y: np.ndarray) -> Tensor:
        """Mean cross-entropy loss (a scalar Tensor) for contexts x (B, k), targets y (B,)."""
        raise NotImplementedError("MLP")
