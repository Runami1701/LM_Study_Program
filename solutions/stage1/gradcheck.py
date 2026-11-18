"""Numerical gradient checking (reference solution)."""

import numpy as np


def gradcheck(f, params: dict, grads: dict, eps: float = 1e-5, n_checks: int = 20,
              rng: np.random.Generator | None = None) -> dict:
    """Compare analytic grads with centered differences on random entries."""
    if rng is None:
        rng = np.random.default_rng(0)
    result = {}
    for name, p in params.items():
        if name not in grads:
            continue
        g = grads[name]
        n = min(n_checks, p.size)
        worst = 0.0
        for flat in rng.choice(p.size, size=n, replace=False):
            idx = np.unravel_index(flat, p.shape)
            old = p[idx]
            p[idx] = old + eps
            f_plus = f()
            p[idx] = old - eps
            f_minus = f()
            p[idx] = old                                  # always restore
            numeric = (f_plus - f_minus) / (2 * eps)
            analytic = float(g[idx])
            denom = max(abs(numeric), abs(analytic), 1e-8)
            worst = max(worst, abs(analytic - numeric) / denom)
        result[name] = worst
    return result
