"""Optimizers for stage 1 (reference solution). Both update params in place."""

import numpy as np


def sgd_step(params: dict, grads: dict, lr: float) -> None:
    """theta <- theta - lr * grad, in place."""
    for name, p in params.items():
        p -= lr * grads[name]           # in-place: the caller's arrays change


class Adam:
    """Adam with bias correction. Call step(grads) once per training step."""

    def __init__(self, params: dict, lr: float = 1e-3, betas: tuple = (0.9, 0.999), eps: float = 1e-8):
        self.params = params
        self.lr = lr
        self.beta1, self.beta2 = betas
        self.eps = eps
        self.t = 0
        self.m = {n: np.zeros_like(p) for n, p in params.items()}
        self.v = {n: np.zeros_like(p) for n, p in params.items()}

    def step(self, grads: dict) -> None:
        self.t += 1
        b1, b2, t = self.beta1, self.beta2, self.t
        for name, p in self.params.items():
            g = grads[name]
            m, v = self.m[name], self.v[name]
            m *= b1
            m += (1 - b1) * g                       # m_t = b1 m_{t-1} + (1-b1) g
            v *= b2
            v += (1 - b2) * (g * g)                 # v_t = b2 v_{t-1} + (1-b2) g^2
            m_hat = m / (1 - b1**t)                 # bias correction
            v_hat = v / (1 - b2**t)
            p -= self.lr * m_hat / (np.sqrt(v_hat) + self.eps)
