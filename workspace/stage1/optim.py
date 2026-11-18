"""Optimizers. Both update the parameter arrays IN PLACE (the caller's dict of
arrays changes; nothing is returned).

Lesson s1-l10: sgd_step   ->  python -m course check s1-l10
Lesson s1-l11: Adam       ->  python -m course check s1-l11
"""

import numpy as np


def sgd_step(params: dict, grads: dict, lr: float) -> None:
    """theta <- theta - lr * grad for every array in params, in place."""
    raise NotImplementedError("sgd_step")


class Adam:
    """Adam with bias correction.

    Usage:
        opt = Adam(params, lr=3e-3)
        for each step: ...compute grads...; opt.step(grads)

    Attributes the checks read: lr, t (number of steps taken so far),
    m and v (dicts name -> array, same shapes as params).
    `lr` may be changed between steps (learning-rate decay).
    """

    def __init__(self, params: dict, lr: float = 1e-3, betas: tuple = (0.9, 0.999), eps: float = 1e-8):
        raise NotImplementedError("Adam")

    def step(self, grads: dict) -> None:
        """One Adam update of every parameter, in place."""
        raise NotImplementedError("Adam")
