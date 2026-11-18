"""Lesson s2-l08 — learning-rate schedule.  ->  python -m course check s2-l08"""

import math


def lr_at(step: int, warmup: int, max_steps: int, lr_max: float, lr_min: float) -> float:
    """Linear warmup, then cosine decay.

    step < warmup:            lr_max * (step + 1) / warmup
    warmup <= step < max:     lr_min + 0.5 (lr_max - lr_min)(1 + cos(pi * progress)),
                              progress = (step - warmup) / (max_steps - warmup)
    step >= max_steps:        lr_min
    """
    raise NotImplementedError("lr_at")
