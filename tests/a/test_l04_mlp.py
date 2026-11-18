"""Checks for a-l04 — Rebuild the MLP on the engine."""

import numpy as np
import pytest

from tests.helpers import assert_close, load

NAMES = ("C", "W1", "b1", "W2", "b2")


@pytest.mark.fn("MLP")
def test_matches_stage1_loss_and_gradients():
    stage1 = load("stage1.model")
    MLP = load("autograd.mlp").MLP
    rng = np.random.default_rng(0)
    p = stage1.init_params(65, 8, 24, 64, rng)
    x, y = rng.integers(0, 65, (16, 8)), rng.integers(0, 65, 16)
    loss1, cache = stage1.forward(p, x, y)
    g1 = stage1.backward(cache)
    model = MLP({k: v.copy() for k, v in p.items()})
    params = model.parameters()
    assert set(params) == set(NAMES), f"parameters() should have keys {NAMES}"
    loss = model(x, y)
    assert loss.data.shape == (), "the model should return a scalar Tensor"
    assert abs(float(loss.data) - loss1) < 1e-5, f"loss {float(loss.data):.6f} differs from your stage-1 model's {loss1:.6f}"
    loss.backward()
    for name in NAMES:
        assert_close(params[name].grad, g1[name], f"grad of {name}", rtol=1e-5, atol=1e-7,
                     hint="the engine's gradient must equal your hand-written stage-1 gradient")


@pytest.mark.fn("MLP")
def test_training_steps_lower_the_loss():
    stage1 = load("stage1.model")
    MLP = load("autograd.mlp").MLP
    rng = np.random.default_rng(1)
    model = MLP(stage1.init_params(20, 3, 8, 32, rng))
    x, y = rng.integers(0, 20, (32, 3)), rng.integers(0, 20, 32)
    first = None
    for _ in range(100):
        for t in model.parameters().values():
            t.zero_grad()
        loss = model(x, y)
        first = first if first is not None else float(loss.data)
        loss.backward()
        for t in model.parameters().values():
            t.data -= 0.5 * t.grad
    assert float(loss.data) < first - 1.0, f"100 steps of gradient descent only moved the loss from {first:.3f} to {float(loss.data):.3f}"
