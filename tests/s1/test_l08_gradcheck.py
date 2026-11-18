"""Checks for s1-l08 — Gradient checking."""

import numpy as np
import pytest

from tests.helpers import load

MOD = "stage1.gradcheck"


def problem():
    rng = np.random.default_rng(0)
    params = {"W": rng.standard_normal((3, 4)), "b": rng.standard_normal(4)}
    x = rng.standard_normal((5, 3))
    f = lambda: float(np.sum(np.tanh(x @ params["W"] + params["b"]) ** 2))  # noqa: E731
    t = np.tanh(x @ params["W"] + params["b"])
    g_out = 2 * t * (1 - t ** 2)
    grads = {"W": x.T @ g_out, "b": g_out.sum(axis=0)}
    return f, params, grads


@pytest.mark.fn("gradcheck")
def test_correct_gradients_pass():
    m = load(MOD)
    f, params, grads = problem()
    res = m.gradcheck(f, params, grads, rng=np.random.default_rng(0))
    assert isinstance(res, dict) and set(res) == {"W", "b"}, f"expected a dict with keys W and b, got {res!r}"
    for name, err in res.items():
        assert err < 1e-7, f"correct gradients gave relative error {err:.2e} for {name} — check the centered difference and the relative-error formula"


@pytest.mark.fn("gradcheck")
def test_broken_gradient_is_flagged():
    m = load(MOD)
    f, params, grads = problem()
    grads["W"] = grads["W"] * 1.1
    res = m.gradcheck(f, params, grads, rng=np.random.default_rng(0))
    assert res["W"] > 1e-2, f"a gradient off by 10% must show a large relative error (~0.09), got {res['W']:.2e}"
    assert res["b"] < 1e-7, "b's gradient is still correct and should pass"


@pytest.mark.fn("gradcheck")
def test_single_wrong_entry_in_small_array_is_found():
    m = load(MOD)
    f, params, grads = problem()
    grads["b"] = grads["b"].copy()
    grads["b"][2] += 1.0
    res = m.gradcheck(f, params, grads, n_checks=20, rng=np.random.default_rng(0))
    assert res["b"] > 1e-2, "with only 4 entries, every entry should be checked — the wrong one was missed"


@pytest.mark.fn("gradcheck")
def test_params_restored_and_call_count():
    m = load(MOD)
    f, params, grads = problem()
    before = {k: v.copy() for k, v in params.items()}
    calls = {"n": 0}

    def counted():
        calls["n"] += 1
        return f()

    m.gradcheck(counted, params, grads, n_checks=5, rng=np.random.default_rng(0))
    for k in params:
        assert np.array_equal(params[k], before[k]), f"params[{k!r}] changed — restore each entry after perturbing it"
    assert calls["n"] == 2 * (5 + 4), (
        f"expected 2 calls of f per checked entry (5 of W, all 4 of b) = 18, got {calls['n']}"
    )
