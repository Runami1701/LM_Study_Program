"""Lesson s2-l08 — AdamW and gradient clipping.  ->  python -m course check s2-l08"""

import numpy as np


class AdamW:
    """Adam (with bias correction, as in s1-l11) plus DECOUPLED weight decay.

    __init__(named_params: dict name -> Tensor, lr, betas=(0.9, 0.95), eps=1e-8, weight_decay=0.1)
    Attributes the checks read: lr, t, m, v (dicts name -> array).

    step(): uses each parameter's .grad. For parameters with data.ndim >= 2 (weight
    matrices, embeddings) first shrink: data -= lr * weight_decay * data; then apply
    the Adam update. Vectors (biases, LayerNorm gamma/beta) are not decayed.
    """

    def __init__(self, named_params: dict, lr: float = 1e-3, betas=(0.9, 0.95), eps: float = 1e-8,
                 weight_decay: float = 0.1):
        raise NotImplementedError("AdamW")

    def step(self) -> None:
        raise NotImplementedError("AdamW")


def clip_grad_norm(params: list, max_norm: float) -> float:
    """If the global L2 norm of all grads (sqrt of the sum of squares of every entry
    of every grad) exceeds max_norm, scale every grad IN PLACE by max_norm / (norm + 1e-6).

    Returns:
        The norm before clipping, as a Python float.
    """
    raise NotImplementedError("clip_grad_norm")
