"""Test-side helpers for the autograd lessons (independent of the learner's backward)."""

import numpy as np


def run_backward(out, upstream):
    """Topologically sort the graph under `out` and call every _backward, with
    out.grad = upstream. Used to test ops before a-l03's backward exists."""
    topo, seen = [], set()

    def visit(n):
        if id(n) in seen:
            return
        seen.add(id(n))
        for p in n._prev:
            visit(p)
        topo.append(n)

    visit(out)
    for n in topo:
        if n._prev:
            n.grad = np.zeros_like(n.data)
    out.grad = np.asarray(upstream, dtype=out.data.dtype) * np.ones_like(out.data)
    for n in reversed(topo):
        n._backward()


def check_op(T, make, inputs, rng, name, tol=1e-6):
    """Gradcheck an op. make(*tensors) -> Tensor; inputs: list of float64 arrays."""
    from tests.helpers import numerical_grad, rel_error

    tensors = [T(x) for x in inputs]
    out = make(*tensors)
    G = rng.standard_normal(out.data.shape)
    run_backward(out, G)
    for i, (t, x) in enumerate(zip(tensors, inputs)):
        num = numerical_grad(lambda: float(np.sum(make(*[T(v) for v in inputs]).data * G)), x)
        err = rel_error(t.grad, num)
        assert t.grad.shape == x.shape, f"{name}: grad of input {i} has shape {t.grad.shape}, expected {x.shape}"
        assert err < tol, f"{name}: gradient of input {i} is wrong (max relative error {err:.2e})"
