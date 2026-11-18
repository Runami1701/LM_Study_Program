"""The stage-1 MLP rebuilt on the autograd engine: forward pass only (reference)."""

import numpy as np

from . import ops
from .tensor import Tensor


class MLP:
    """Same model as stage 1. Build it from a params dict (e.g. stage-1 init_params)."""

    def __init__(self, params: dict):
        self.C = Tensor(params["C"])
        self.W1 = Tensor(params["W1"])
        self.b1 = Tensor(params["b1"])
        self.W2 = Tensor(params["W2"])
        self.b2 = Tensor(params["b2"])

    def parameters(self) -> dict:
        return {"C": self.C, "W1": self.W1, "b1": self.b1, "W2": self.W2, "b2": self.b2}

    def __call__(self, x: np.ndarray, y: np.ndarray) -> Tensor:
        B, k = x.shape
        emb = self.C[x]                                  # (B, k, e), via getitem
        flat = ops.reshape(emb, (B, -1))                 # (B, k*e)
        h = ops.tanh(flat @ self.W1 + self.b1)           # (B, h)
        logits = h @ self.W2 + self.b2                   # (B, V)
        return ops.cross_entropy(logits, y)
