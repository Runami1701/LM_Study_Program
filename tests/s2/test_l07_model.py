"""Checks for s2-l07 — Transformer block + full model."""

import numpy as np
import pytest

from tests.helpers import assert_close, check_array, load
from tests.s2.common import T, grad_check_module, to64


@pytest.mark.fn("MLP")
def test_mlp():
    blk = load("stage2.block")
    mlp = blk.MLP(8, np.random.default_rng(0))
    assert mlp.fc.W.data.shape == (8, 32) and mlp.proj.W.data.shape == (32, 8), "MLP: Linear(d, 4d) then Linear(4d, d)"
    x = T()(np.random.default_rng(1).standard_normal((2, 3, 8)).astype(np.float32))
    out = mlp(x)
    check_array(out.data, "MLP output", shape=(2, 3, 8), dtype=np.float32)


@pytest.mark.fn("Block")
def test_block_residuals():
    blk = load("stage2.block")
    b = blk.Block(8, 2, np.random.default_rng(0))
    names = set(b.named_parameters())
    for n in ("ln1.gamma", "attn.qkv.W", "ln2.beta", "mlp.fc.W"):
        assert n in names, f"Block should contain {n} (attributes ln1, attn, ln2, mlp)"
    b.attn.proj.W.data[...] = 0
    b.attn.proj.b.data[...] = 0
    b.mlp.proj.W.data[...] = 0
    b.mlp.proj.b.data[...] = 0
    x = np.random.default_rng(1).standard_normal((2, 4, 8)).astype(np.float32)
    assert_close(b(T()(x)).data, x, "Block(x) with zeroed output projections",
                 hint="pre-LN residual block: x + attn(ln1(x)), then x + mlp(ln2(x)); with zero branches it is the identity")


@pytest.mark.fn("Block")
def test_block_gradcheck():
    blk = load("stage2.block")
    ops = load("autograd.ops")
    rng = np.random.default_rng(2)
    b = blk.Block(4, 2, np.random.default_rng(3))
    to64(b)
    for p in b.parameters():
        p.data[...] += rng.standard_normal(p.data.shape) * 0.3
    x = T()(rng.standard_normal((1, 3, 4)))
    G = rng.standard_normal((1, 3, 4))
    params = b.named_parameters()
    pick = {k: params[k] for k in ("ln1.gamma", "attn.qkv.W", "mlp.fc.W", "mlp.proj.b")}
    grad_check_module(lambda: ops.sum(b(x) * G), {"x": x, **pick}, n=5, tol=1e-5)


@pytest.mark.fn("GPT")
def test_param_count_and_initial_loss():
    mod = load("stage2.model")
    model = mod.GPT(65, 128, 128, 4, 4, np.random.default_rng(0))
    n = mod.count_params(model)
    assert isinstance(n, int), "count_params returns a Python int"
    assert n == 826_368, (
        f"V=65, T=128, d=128, 4 heads, 4 layers has 4·(12d² + 13d) + 2·Vd + Td + 2d = 826,368 parameters, got {n:,}"
    )
    rng = np.random.default_rng(1)
    x = rng.integers(0, 65, (4, 32))
    logits, loss = model(x, rng.integers(0, 65, (4, 32)))
    check_array(logits.data, "logits", shape=(4, 32, 65), dtype=np.float32, shape_desc="(B, t, V)")
    assert loss.data.shape == (), "loss must be a scalar Tensor"
    assert abs(float(loss.data) - np.log(65)) < 0.15, f"initial loss {float(loss.data):.3f} should be near ln 65 = 4.17"
    _, none = model(x)
    assert none is None, "without targets the loss is None"


@pytest.mark.fn("GPT")
def test_overfit_one_batch():
    mod = load("stage2.model")
    rng = np.random.default_rng(2)
    model = mod.GPT(10, 8, 16, 2, 1, np.random.default_rng(3))
    x = rng.integers(0, 10, (4, 8))
    y = rng.integers(0, 10, (4, 8))
    params = model.parameters()
    m = [np.zeros_like(p.data) for p in params]
    v = [np.zeros_like(p.data) for p in params]
    loss = None
    for t in range(1, 301):
        model.zero_grad()
        _, loss = model(x, y)
        loss.backward()
        for p, mi, vi in zip(params, m, v):
            mi[...] = 0.9 * mi + 0.1 * p.grad
            vi[...] = 0.999 * vi + 0.001 * p.grad ** 2
            p.data -= 1e-2 * (mi / (1 - 0.9 ** t)) / (np.sqrt(vi / (1 - 0.999 ** t)) + 1e-8)
        if float(loss.data) < 0.05:
            return
    raise AssertionError(f"a tiny GPT could not memorize 32 tokens in 300 Adam steps (loss {float(loss.data):.3f}) — some gradient is wrong")
