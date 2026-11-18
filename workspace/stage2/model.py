"""The character-level GPT.

Lesson s2-l07: GPT, count_params                    ->  python -m course check s2-l07
Lesson s2-l09: save_checkpoint, load_checkpoint     ->  python -m course check s2-l09
"""

import numpy as np

from ..autograd import ops
from .block import Block
from .layers import Embeddings, LayerNorm, Linear, Module


class GPT(Module):
    """Create, in this order: self.T = T; self.emb = Embeddings(V, T, d, rng);
    self.blocks = [Block(d, n_heads, rng) for each layer]; self.ln_f = LayerNorm(d);
    self.head = Linear(d, V, rng, bias=False).
    """

    def __init__(self, V: int, T: int, d: int, n_heads: int, n_layers: int, rng: np.random.Generator):
        raise NotImplementedError("GPT")

    def __call__(self, idx: np.ndarray, targets: np.ndarray | None = None):
        """idx (B, t) ints with t <= T -> (logits Tensor (B, t, V), loss).

        loss is the mean cross-entropy over all B*t positions (a scalar Tensor),
        or None when targets is None.
        """
        raise NotImplementedError("GPT")


def count_params(model: Module) -> int:
    """Total number of scalars in all parameters (Python int)."""
    raise NotImplementedError("count_params")


def save_checkpoint(model: Module, path) -> None:
    """Save every parameter's data under its name from named_parameters(), in one .npz file."""
    raise NotImplementedError("save_checkpoint")


def load_checkpoint(model: Module, path) -> None:
    """Copy the arrays from a .npz written by save_checkpoint into the model's parameters,
    IN PLACE (p.data[...] = saved). Raise KeyError if a name is missing and ValueError
    if a shape differs."""
    raise NotImplementedError("load_checkpoint")
