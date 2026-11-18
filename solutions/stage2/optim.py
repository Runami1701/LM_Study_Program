"""AdamW and gradient clipping for Tensors (reference solution)."""

import numpy as np


class AdamW:
    """Adam with decoupled weight decay, applied only to tensors with ndim >= 2."""

    def __init__(self, named_params: dict, lr: float = 1e-3, betas=(0.9, 0.95), eps: float = 1e-8,
                 weight_decay: float = 0.1):
        self.params = named_params
        self.lr = lr
        self.beta1, self.beta2 = betas
        self.eps = eps
        self.weight_decay = weight_decay
        self.t = 0
        self.m = {n: np.zeros_like(p.data) for n, p in named_params.items()}
        self.v = {n: np.zeros_like(p.data) for n, p in named_params.items()}

    def step(self) -> None:
        self.t += 1
        b1, b2, t = self.beta1, self.beta2, self.t
        for name, p in self.params.items():
            g = p.grad
            m, v = self.m[name], self.v[name]
            m *= b1
            m += (1 - b1) * g
            v *= b2
            v += (1 - b2) * (g * g)
            m_hat = m / (1 - b1**t)
            v_hat = v / (1 - b2**t)
            if p.data.ndim >= 2 and self.weight_decay:
                p.data -= self.lr * self.weight_decay * p.data     # decoupled: not part of the gradient
            p.data -= self.lr * m_hat / (np.sqrt(v_hat) + self.eps)


def clip_grad_norm(params: list, max_norm: float) -> float:
    """Scale all grads so their global L2 norm is at most max_norm. Returns the norm before clipping."""
    total = float(np.sqrt(sum(float((p.grad.astype(np.float64) ** 2).sum()) for p in params)))
    scale = min(1.0, max_norm / (total + 1e-6))
    if scale < 1.0:
        for p in params:
            p.grad *= scale
    return total
