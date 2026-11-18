"""Checks for s1-l09 — Assemble the MLP, initialize, overfit one batch."""

import numpy as np
import pytest

from tests.helpers import assert_close, check_array, load, numerical_grad, rel_error

MOD = "stage1.model"
NAMES = ("C", "W1", "b1", "W2", "b2")


def ce64(logits, y):
    z = logits - logits.max(axis=1, keepdims=True)
    lse = np.log(np.exp(z).sum(axis=1))
    return float(np.mean(lse - z[np.arange(len(y)), y]))


@pytest.mark.fn("init_params")
def test_init_shapes_dtypes_and_draw_order():
    m = load(MOD)
    p = m.init_params(65, 8, 24, 512, np.random.default_rng(0))
    assert isinstance(p, dict) and set(p) == set(NAMES), f"expected keys {NAMES}, got {sorted(p) if isinstance(p, dict) else p}"
    shapes = {"C": (65, 24), "W1": (192, 512), "b1": (512,), "W2": (512, 65), "b2": (65,)}
    for name, shape in shapes.items():
        check_array(p[name], name, shape=shape, dtype=np.float32)
    rng = np.random.default_rng(0)
    C = rng.standard_normal((65, 24))
    W1 = rng.standard_normal((192, 512)) / np.sqrt(192)
    W2 = rng.standard_normal((512, 65)) * 0.01
    assert_close(p["C"], C, "C", hint="draw C first, with rng.standard_normal((V, e))")
    assert_close(p["W1"], W1, "W1", hint="then W1 = rng.standard_normal((k*e, h)) / sqrt(k*e)")
    assert_close(p["W2"], W2, "W2", hint="then W2 = rng.standard_normal((h, V)) * 0.01")
    assert not p["b1"].any() and not p["b2"].any(), "biases start at zero"


@pytest.mark.fn("count_params")
def test_count_params():
    m = load(MOD)
    p = m.init_params(65, 8, 24, 512, np.random.default_rng(0))
    n = m.count_params(p)
    assert isinstance(n, int), f"count_params should return a Python int, got {type(n).__name__}"
    assert n == 133_721, f"V=65, k=8, e=24, h=512 has 1,560 + 98,304 + 512 + 33,280 + 65 = 133,721 params, got {n:,}"


@pytest.mark.fn("predict")
def test_predict_shape_and_consistency():
    m = load(MOD)
    rng = np.random.default_rng(1)
    p = m.init_params(11, 3, 4, 6, rng)
    x = rng.integers(0, 11, (5, 3))
    logits = m.predict(p, x)
    check_array(logits, "logits", shape=(5, 11), dtype=np.float32, shape_desc="(B, V)")
    flat = p["C"][x].reshape(5, -1).astype(np.float64)
    expected = np.tanh(flat @ p["W1"] + p["b1"]) @ p["W2"] + p["b2"]
    assert_close(logits, expected, "logits", atol=1e-5)


@pytest.mark.fn("forward")
def test_initial_loss_near_ln_V():
    m = load(MOD)
    rng = np.random.default_rng(2)
    p = m.init_params(65, 8, 24, 512, rng)
    x, y = rng.integers(0, 65, (256, 8)), rng.integers(0, 65, 256)
    loss, _ = m.forward(p, x, y)
    assert isinstance(loss, float), f"forward should return the loss as a Python float, got {type(loss).__name__}"
    assert abs(loss - np.log(65)) < 0.15, (
        f"initial loss {loss:.3f} should be within 0.15 of ln 65 = 4.174 — is W2 scaled by 0.01?"
    )
    assert_close(loss, ce64(m.predict(p, x).astype(np.float64), y), "forward loss vs predict", atol=1e-4)


@pytest.mark.fn("backward")
def test_full_model_gradcheck():
    m = load(MOD)
    rng = np.random.default_rng(3)
    p = {k: v.astype(np.float64) for k, v in m.init_params(11, 3, 4, 6, rng).items()}
    p["W2"] *= 30  # larger output weights make every gradient clearly non-zero
    x = np.array([[1, 2, 2], [0, 5, 10], [3, 3, 3], [7, 1, 0]])
    y = np.array([2, 4, 3, 10])
    loss, cache = m.forward(p, x, y)
    grads = m.backward(cache)
    assert isinstance(grads, dict) and set(grads) == set(NAMES), f"backward should return a dict with keys {NAMES}"
    for name in NAMES:
        check_array(grads[name], f"grads[{name!r}]", shape=p[name].shape)
        num = numerical_grad(lambda: m.forward(p, x, y)[0], p[name])
        err = rel_error(grads[name], num)
        assert err < 1e-5, f"grads[{name!r}] differs from the numerical gradient (max relative error {err:.2e})"


@pytest.mark.fn("backward")
def test_overfit_one_batch():
    m = load(MOD)
    rng = np.random.default_rng(4)
    p = m.init_params(65, 8, 24, 128, rng)
    x, y = rng.integers(0, 65, (32, 8)), rng.integers(0, 65, 32)
    loss = None
    for step in range(1000):
        loss, cache = m.forward(p, x, y)
        if loss < 0.05:
            return
        g = m.backward(cache)
        for name in p:
            p[name] -= 0.5 * g[name]
    raise AssertionError(
        f"plain gradient descent (lr 0.5) could not overfit 32 examples in 1000 steps (loss {loss:.3f}) — "
        "a correct model memorizes them easily, so some gradient is wrong"
    )
