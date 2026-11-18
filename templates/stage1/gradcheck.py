"""Lesson s1-l08 — Gradient checking.  ->  python -m course check s1-l08"""

import numpy as np


def gradcheck(f, params: dict, grads: dict, eps: float = 1e-5, n_checks: int = 20,
              rng: np.random.Generator | None = None) -> dict:
    """Spot-check analytic gradients against centered differences.

    Args:
        f:        a function with no arguments that returns the scalar loss, computed
                  from the arrays in `params` (so changing params[name] in place changes f()).
        params:   dict name -> float64 array. Perturb entries in place, and always restore them.
        grads:    dict name -> analytic gradient array (same shape as params[name]).
                  Names missing from grads are skipped.
        eps:      the step for the centered difference (f(p + eps) - f(p - eps)) / (2 eps).
        n_checks: how many random entries to check per array (all of them if the array is smaller).
        rng:      Generator used to choose the entries (default: np.random.default_rng(0)).

    Returns:
        dict name -> the largest relative error |a - n| / max(|a|, |n|, 1e-8) over the
        checked entries, where a is analytic and n numeric.
    """
    raise NotImplementedError("gradcheck")
