"""Learning-rate schedule (reference solution)."""

import math


def lr_at(step: int, warmup: int, max_steps: int, lr_max: float, lr_min: float) -> float:
    """Linear warmup to lr_max over `warmup` steps, then cosine decay to lr_min at max_steps."""
    if step < warmup:
        return lr_max * (step + 1) / warmup
    if step >= max_steps:
        return lr_min
    progress = (step - warmup) / (max_steps - warmup)          # 0 -> 1
    return lr_min + 0.5 * (lr_max - lr_min) * (1.0 + math.cos(math.pi * progress))
