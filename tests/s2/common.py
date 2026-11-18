"""Test-side helpers for stage 2."""

import numpy as np

from tests.helpers import load, numerical_grad, rel_error


def T():
    return load("autograd.tensor").Tensor


def to64(module):
    for p in module.parameters():
        p.data = p.data.astype(np.float64)
        p.grad = np.zeros_like(p.data)


def grad_check_module(f_loss, tensors: dict, n: int = 6, tol: float = 1e-5, rng=None):
    """f_loss() -> scalar Tensor. Compares .grad (after one backward) with centered
    differences on up to n random entries of each tensor."""
    rng = rng or np.random.default_rng(0)
    for t in tensors.values():
        t.grad = np.zeros_like(t.data)
    loss = f_loss()
    loss.backward()
    for name, t in tensors.items():
        flat = rng.choice(t.data.size, size=min(n, t.data.size), replace=False)
        for fi in flat:
            idx = np.unravel_index(fi, t.data.shape)
            old = t.data[idx]
            t.data[idx] = old + 1e-5
            fp = float(f_loss().data)
            t.data[idx] = old - 1e-5
            fm = float(f_loss().data)
            t.data[idx] = old
            num = (fp - fm) / 2e-5
            ana = float(t.grad[idx])
            err = abs(num - ana) / max(abs(num), abs(ana), 1e-8)
            assert err < tol, f"gradient of {name}{list(idx)} is wrong: analytic {ana:.6g}, numerical {num:.6g}"


def ref_layernorm(x, g, b, eps=1e-5):
    mu = x.mean(-1, keepdims=True)
    var = x.var(-1, keepdims=True)
    return (x - mu) / np.sqrt(var + eps) * g + b


def ref_softmax(z):
    z = z - z.max(-1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(-1, keepdims=True)


def ref_gelu(x):
    return 0.5 * x * (1 + np.tanh(np.sqrt(2 / np.pi) * (x + 0.044715 * x**3)))
