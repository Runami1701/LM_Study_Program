"""Stage 2 building blocks, written on YOUR autograd engine (workspace/autograd).

Lesson s2-l01: get_batch_seq, Linear, Embeddings   ->  python -m course check s2-l01
Lesson s2-l02: layernorm, LayerNorm                ->  python -m course check s2-l02
Lesson s2-l03: gelu                                ->  python -m course check s2-l03

Every model array is float32. New ops (layernorm, gelu) follow the a-l02 pattern:
compute the output with NumPy, wrap it in a Tensor with its parents, attach a
_backward closure that ACCUMULATES into each parent's grad.
"""

import math

import numpy as np

from ..autograd import ops
from ..autograd.tensor import Tensor, unbroadcast


class Module:
    """Given: collects Tensor attributes (and sub-modules, also inside lists).

    named_parameters() -> {"blocks.0.attn.qkv.W": Tensor, ...}; parameters() -> list;
    zero_grad() zeroes every parameter's grad.
    """

    def named_parameters(self, prefix: str = "") -> dict:
        out = {}
        for name, value in vars(self).items():
            if isinstance(value, Tensor):
                out[prefix + name] = value
            elif isinstance(value, Module):
                out.update(value.named_parameters(prefix + name + "."))
            elif isinstance(value, (list, tuple)):
                for i, item in enumerate(value):
                    if isinstance(item, Module):
                        out.update(item.named_parameters(f"{prefix}{name}.{i}."))
        return out

    def parameters(self) -> list:
        return list(self.named_parameters().values())

    def zero_grad(self) -> None:
        for p in self.parameters():
            p.zero_grad()


def get_batch_seq(data: np.ndarray, B: int, T: int, rng: np.random.Generator):
    """Sample B random windows of T+1 consecutive ids.

    Draw starts with exactly: rng.integers(0, len(data) - T, size=B).

    Returns:
        x: (B, T) int64, x[r] = data[s_r : s_r + T]
        y: (B, T) int64, y[r] = data[s_r + 1 : s_r + T + 1]   (the next id at every position)
    """
    raise NotImplementedError("get_batch_seq")


class Linear(Module):
    """out = x @ W + b, for x of shape (..., n_in).

    __init__ draws W = rng.standard_normal((n_in, n_out)) * 0.02 (float32), b = zeros(n_out),
    as Tensors stored in self.W and self.b. With bias=False, set self.b = None.
    """

    def __init__(self, n_in: int, n_out: int, rng: np.random.Generator, bias: bool = True):
        raise NotImplementedError("Linear")

    def __call__(self, x: Tensor) -> Tensor:
        raise NotImplementedError("Linear")


class Embeddings(Module):
    """Token + learned position embeddings.

    __init__: self.wte = Tensor (V, d) and then self.wpe = Tensor (T, d), both
    rng.standard_normal(...) * 0.02 as float32, drawn in that order.
    __call__(idx (B, t) ints, t <= T) -> Tensor (B, t, d): token vectors plus the
    position vectors of positions 0..t-1.
    """

    def __init__(self, V: int, T: int, d: int, rng: np.random.Generator):
        raise NotImplementedError("Embeddings")

    def __call__(self, idx: np.ndarray) -> Tensor:
        raise NotImplementedError("Embeddings")


def layernorm(x: Tensor, gamma: Tensor, beta: Tensor, eps: float = 1e-5) -> Tensor:
    """Layer normalization over the last axis, as ONE op with a derived backward.

    x: (..., d); gamma, beta: (d,).  out = (x - mean) / sqrt(var + eps) * gamma + beta,
    with mean and (biased) var over the last axis. The backward must fill x.grad,
    gamma.grad and beta.grad.
    """
    raise NotImplementedError("layernorm")


class LayerNorm(Module):
    """self.gamma = ones(d), self.beta = zeros(d) (float32 Tensors); calls layernorm."""

    def __init__(self, d: int):
        raise NotImplementedError("LayerNorm")

    def __call__(self, x: Tensor) -> Tensor:
        raise NotImplementedError("LayerNorm")


def gelu(x: Tensor) -> Tensor:
    """GELU (tanh approximation), as one op:
        out = 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x**3)))
    Keep float32: build constants as Python floats (math.sqrt), not NumPy scalars.
    """
    raise NotImplementedError("gelu")
