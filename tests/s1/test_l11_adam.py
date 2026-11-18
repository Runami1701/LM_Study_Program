"""Checks for s1-l11 — Adam."""

import numpy as np
import pytest

from tests.helpers import assert_close, load


def adam_reference(theta, grads, lr=0.1, b1=0.9, b2=0.999, eps=1e-8):
    theta = np.array(theta, dtype=np.float64)
    m = np.zeros_like(theta)
    v = np.zeros_like(theta)
    out = []
    for t, g in enumerate(grads, start=1):
        g = np.asarray(g, dtype=np.float64)
        m = b1 * m + (1 - b1) * g
        v = b2 * v + (1 - b2) * g * g
        theta = theta - lr * (m / (1 - b1**t)) / (np.sqrt(v / (1 - b2**t)) + eps)
        out.append(theta.copy())
    return out


@pytest.mark.fn("Adam")
def test_three_steps_by_hand():
    m = load("stage1.optim")
    p = np.array([1.0, -2.0], dtype=np.float32)
    opt = m.Adam({"p": p}, lr=0.1)
    gs = [[0.5, -1.0], [0.1, 3.0], [-0.2, 0.0]]
    expected = adam_reference([1.0, -2.0], gs)
    for step, g in enumerate(gs):
        opt.step({"p": np.array(g, dtype=np.float32)})
        assert_close(p, expected[step], f"p after step {step + 1}", rtol=1e-5, atol=1e-6,
                     hint="update p in place; with bias correction, the first step moves each entry by about lr")


@pytest.mark.fn("Adam")
def test_first_step_size_is_lr():
    m = load("stage1.optim")
    p = np.array([0.0, 0.0, 0.0], dtype=np.float32)
    opt = m.Adam({"p": p}, lr=0.01)
    opt.step({"p": np.array([1e-4, 5.0, -300.0], dtype=np.float32)})
    assert_close(np.abs(p), [0.01, 0.01, 0.01], "|first step|", rtol=1e-3,
                 hint="with bias correction m_hat / sqrt(v_hat) = g / |g| at t = 1, whatever the gradient's size")


@pytest.mark.fn("Adam")
def test_state_and_lr_attribute():
    m = load("stage1.optim")
    params = {"W": np.ones((2, 3), np.float32), "b": np.zeros(3, np.float32)}
    opt = m.Adam(params, lr=0.05)
    assert opt.lr == 0.05 and opt.t == 0, "after construction: lr as given and t = 0"
    for _ in range(2):
        opt.step({"W": np.ones((2, 3), np.float32), "b": np.ones(3, np.float32)})
    assert opt.t == 2, f"t should count steps taken (2), got {opt.t}"
    for name in params:
        assert opt.m[name].shape == params[name].shape and opt.v[name].shape == params[name].shape, (
            f"m and v for {name} must have the parameter's shape"
        )
    W_before = params["W"].copy()
    opt.lr = 0.0
    opt.step({"W": np.ones((2, 3), np.float32), "b": np.ones(3, np.float32)})
    assert np.array_equal(params["W"], W_before), "setting opt.lr = 0 must stop updates (read self.lr in step)"
