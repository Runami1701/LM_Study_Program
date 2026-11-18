---
id: a-l04
title: Rebuild the MLP on the engine
stage: a
order: 4
prereqs: [a-l03]
files: [workspace/autograd/mlp.py]
functions: [MLP]
tests: tests/a/test_l04_mlp.py
est_minutes: 30
---

::: goal
Write the stage-1 model again as a forward pass only, on your engine, and confirm that its loss and **every** gradient equal your hand-written stage-1 backward pass.
:::

::: notation
| Symbol | Meaning |
| --- | --- |
| $C, W_1, b_1, W_2, b_2$ | the stage-1 parameters, now wrapped in Tensors |
| $B, k, e, h, V$ | batch, context, embedding, hidden and vocabulary sizes |
| $x, y$ | contexts `(B, k)` and targets `(B,)`, plain integer arrays |
| $L$ | the loss, a scalar Tensor |
| rtol | relative tolerance of the comparison |
:::

::: concept
### The same model, a quarter of the code

| step | stage 1 (manual) | engine |
| --- | --- | --- |
| embed | `embedding_forward` + `embedding_backward` | `C[x]` |
| flatten | `flatten_context` + reshape back | `ops.reshape(emb, (B, -1))` |
| linear | `linear_forward` + `linear_backward` | `flat @ W1 + b1` |
| tanh | `tanh_forward` + `tanh_backward` | `ops.tanh(...)` |
| output | `linear_forward` + `linear_backward` | `h @ W2 + b2` |
| loss | `cross_entropy` + `cross_entropy_backward` | `ops.cross_entropy(logits, y)` |
| backward | hand-ordered calls in `backward(cache)` | `loss.backward()` |

No caches, no reverse ordering. The bias broadcast is handled by `unbroadcast`, the duplicate ids in `C[x]` by `getitem`'s `np.add.at`.

### Why the gradients must match exactly

Both versions compute the same mathematical function with the same formulas: your engine's ops are the stage-1 derivations moved into closures. So with identical parameters and inputs, the loss and gradients agree up to float32 rounding (the check uses a relative tolerance of $10^{-5}$). If they don't, one of the two is wrong, and you have a perfect test oracle: two independent implementations that must agree.

### The training step on the engine

Each step:

1. call `zero_grad()` on every parameter Tensor;
2. `loss = model(x, y)`;
3. `loss.backward()`;
4. for every parameter, `data -= lr * grad`.

(The optimizers of stage 2 will work on Tensors directly.) The `zero_grad` line is new: the manual backward returned fresh arrays every step, but the engine accumulates into leaves.

### What this buys for stage 2

A transformer block has layer norms, attention with reshapes and transposes, residual additions and an MLP. Writing its backward by hand would be long and fragile. On the engine, a block is ~10 lines of forward code; only genuinely new operations (LayerNorm, GELU, softmax) need derived backwards, each checked once in isolation.
:::

::: think
Q: The engine's gradient for `b1` matches stage 1 but `C` doesn't. Which engine op is the first suspect?
A: `getitem` (the lookup `C[x]`): its backward must use `np.add.at` so repeated ids accumulate. A buffered `grad[key] += g` would match whenever ids in the batch happen to be unique, and fail otherwise.
:::

::: think
Q: Why compare against your own stage-1 code instead of finite differences?
A: Both are valid. Stage 1 has already been gradient-checked, so it is a cheap, exact oracle for the whole model at once, in float32, with no need for float64 copies or two forward passes per entry.
:::

::: hint level=1
`__init__` wraps the five arrays as Tensors; `__call__` is the forward pass of s1-l09 written with Tensor operations; `parameters` returns the five Tensors by name.
:::

::: hint level=2
`Tensor(params["C"])` and so on. `self.C[x]` is the lookup, `ops.reshape(emb, (B, -1))` flattens, `@` and `+` build the linear layers, `ops.tanh` and `ops.cross_entropy` finish it.
:::

::: hint level=3
```pseudo
__init__(params):  C, W1, b1, W2, b2 ← Tensor of each array
parameters():      return {name: tensor} for the five
__call__(x, y):
    B ← rows of x
    emb  ← C[x]                         # (B, k, e)
    flat ← reshape(emb, (B, −1))        # (B, k·e)
    h    ← tanh(flat @ W1 + b1)         # (B, h)
    logits ← h @ W2 + b2                # (B, V)
    return cross_entropy(logits, y)
```
:::

::: attempt
File: `workspace/autograd/mlp.py` — implement the `MLP` class.
Check: `python -m course check a-l04`
:::

::: solution
@include solutions/autograd/mlp.py#MLP

- The forward pass reads exactly like the table in s1-l09; the engine records the graph as it runs.
- `parameters()` returns the Tensors themselves (not copies), so updating `.data` in place trains the model.
- No backward code anywhere: `loss.backward()` walks the recorded graph.
:::

::: breakit
Train the engine MLP for 50 steps **without** `zero_grad`, printing the norm of `W1.grad` each step. Expect the norm to grow roughly linearly with the step count (each step adds a fresh gradient to all the old ones), and the loss to stop improving or blow up.
:::

::: recap
- The whole stage-1 model is a dozen lines of forward code on your engine.
- Loss and all gradients equal the hand-written version: two independent implementations agree.
- Training on the engine is zero_grad → forward → backward → update.
:::
