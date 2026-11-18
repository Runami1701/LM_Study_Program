"""The character-level GPT (reference solution)."""

import numpy as np

from ..autograd import ops
from .block import Block
from .layers import Embeddings, LayerNorm, Linear, Module


class GPT(Module):
    def __init__(self, V: int, T: int, d: int, n_heads: int, n_layers: int, rng: np.random.Generator):
        self.T = T
        self.emb = Embeddings(V, T, d, rng)
        self.blocks = [Block(d, n_heads, rng) for _ in range(n_layers)]
        self.ln_f = LayerNorm(d)
        self.head = Linear(d, V, rng, bias=False)

    def __call__(self, idx: np.ndarray, targets: np.ndarray | None = None):
        """idx (B, t) with t <= T -> logits (B, t, V) and the mean loss (or None)."""
        B, t = idx.shape
        x = self.emb(idx)                                  # (B, t, d)
        for block in self.blocks:
            x = block(x)
        logits = self.head(self.ln_f(x))                   # (B, t, V)
        if targets is None:
            return logits, None
        V = logits.shape[-1]
        loss = ops.cross_entropy(ops.reshape(logits, (B * t, V)), targets.reshape(-1))
        return logits, loss


def count_params(model: Module) -> int:
    return int(sum(p.data.size for p in model.parameters()))


def save_checkpoint(model: Module, path) -> None:
    """All parameters by name in one .npz file."""
    np.savez(path, **{name: p.data for name, p in model.named_parameters().items()})


def load_checkpoint(model: Module, path) -> None:
    """Copy saved arrays into the model's parameters, in place, checking names and shapes."""
    saved = np.load(path)
    params = model.named_parameters()
    missing = set(params) - set(saved.files)
    if missing:
        raise KeyError(f"checkpoint is missing {sorted(missing)}")
    for name, p in params.items():
        if saved[name].shape != p.data.shape:
            raise ValueError(f"{name}: checkpoint shape {saved[name].shape} != model shape {p.data.shape}")
        p.data[...] = saved[name]
