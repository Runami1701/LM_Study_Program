"""Checks for s2-l08 — AdamW, LR schedule, gradient clipping."""

import math

import numpy as np
import pytest

from tests.helpers import assert_close, load
from tests.s2.common import T


@pytest.mark.fn("AdamW")
def test_adamw_two_steps_by_hand():
    opt_mod = load("stage2.optim")
    Tn = T()
    W = Tn(np.array([[1.0, -2.0]], dtype=np.float32))     # matrix: decayed
    b = Tn(np.array([0.5], dtype=np.float32))             # vector: not decayed
    opt = opt_mod.AdamW({"W": W, "b": b}, lr=0.1, betas=(0.9, 0.95), weight_decay=0.5)
    gW = [np.array([[0.2, 0.4]]), np.array([[-0.1, 0.4]])]
    gb = [np.array([1.0]), np.array([1.0])]
    w, bb = np.array([[1.0, -2.0]]), np.array([0.5])
    mw = np.zeros_like(w); vw = np.zeros_like(w); mb = np.zeros_like(bb); vb = np.zeros_like(bb)
    for t in (1, 2):
        W.grad = gW[t - 1].astype(np.float32); b.grad = gb[t - 1].astype(np.float32)
        opt.step()
        mw = 0.9 * mw + 0.1 * gW[t - 1]; vw = 0.95 * vw + 0.05 * gW[t - 1] ** 2
        mb = 0.9 * mb + 0.1 * gb[t - 1]; vb = 0.95 * vb + 0.05 * gb[t - 1] ** 2
        w = w - 0.1 * 0.5 * w
        w = w - 0.1 * (mw / (1 - 0.9 ** t)) / (np.sqrt(vw / (1 - 0.95 ** t)) + 1e-8)
        bb = bb - 0.1 * (mb / (1 - 0.9 ** t)) / (np.sqrt(vb / (1 - 0.95 ** t)) + 1e-8)
        assert_close(W.data, w, f"W after step {t}", rtol=1e-5,
                     hint="matrices: data -= lr * wd * data (decoupled), then the Adam step")
        assert_close(b.data, bb, f"b after step {t}", rtol=1e-5,
                     hint="vectors (ndim < 2) get no weight decay")
    assert opt.t == 2


@pytest.mark.fn("lr_at")
def test_schedule_shape():
    lr_at = load("stage2.schedule").lr_at
    kw = dict(warmup=10, max_steps=110, lr_max=1e-3, lr_min=1e-4)
    assert math.isclose(lr_at(0, **kw), 1e-4), "step 0 of a 10-step warmup: lr_max * 1/10"
    assert math.isclose(lr_at(9, **kw), 1e-3), "the last warmup step reaches lr_max"
    assert math.isclose(lr_at(10, **kw), 1e-3), "cosine starts at lr_max (progress 0)"
    assert math.isclose(lr_at(60, **kw), 5.5e-4), "halfway through the cosine: (lr_max + lr_min) / 2"
    assert math.isclose(lr_at(110, **kw), 1e-4) and math.isclose(lr_at(500, **kw), 1e-4), "at and after max_steps: lr_min"
    vals = [lr_at(s, **kw) for s in range(10, 111)]
    assert all(a >= b - 1e-15 for a, b in zip(vals, vals[1:])), "the cosine phase must decrease monotonically"


@pytest.mark.fn("clip_grad_norm")
def test_clipping():
    clip = load("stage2.optim").clip_grad_norm
    Tn = T()
    a, b = Tn(np.zeros(2, np.float32)), Tn(np.zeros((1, 1), np.float32))
    a.grad = np.array([3.0, 0.0], np.float32); b.grad = np.array([[4.0]], np.float32)
    n = clip([a, b], 1.0)
    assert isinstance(n, float) and abs(n - 5.0) < 1e-5, f"return the global norm before clipping (sqrt(9 + 16) = 5), got {n}"
    assert_close(a.grad, [0.6, 0.0], "a.grad after clipping to 1", atol=1e-5, hint="scale every grad by max_norm / norm")
    assert_close(b.grad, [[0.8]], "b.grad after clipping to 1", atol=1e-5)
    a.grad = np.array([0.3, 0.0], np.float32); b.grad = np.array([[0.4]], np.float32)
    clip([a, b], 1.0)
    assert_close(a.grad, [0.3, 0.0], "a grad below the threshold", hint="norms below max_norm are left unchanged")
