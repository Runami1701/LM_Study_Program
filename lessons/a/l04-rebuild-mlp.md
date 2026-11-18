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
You will write the Stage 1 model again, this time as a forward pass only, on your engine. Then you confirm that its loss and **every** gradient equal those of your hand-written Stage 1 backward pass.
:::

::: notation
| Symbol | Meaning |
| --- | --- |
| $C, W_1, b_1, W_2, b_2$ | the Stage 1 parameters, now stored in Tensors |
| $B, k, e, h, V$ | batch, context, embedding, hidden and vocabulary sizes |
| $x, y$ | contexts `(B, k)` and correct ids `(B,)`, as plain integer arrays |
| $L$ | the loss, a Tensor holding one number |
:::

::: numpy
### Tensors behave like arrays in expressions

Your Tensor class defines `+`, `*`, `@` and indexing, so model code on the engine reads like NumPy code:

```python
emb = self.C[x]                         # lookup (getitem)
flat = ops.reshape(emb, (B, -1))        # reshape
h = ops.tanh(flat @ self.W1 + self.b1)  # matmul, add with broadcasting, tanh
```

Each line builds new nodes of the graph. `loss.backward()` later walks back through all of them.

### Reading the values

A Tensor's numbers are in `.data` and its gradient in `.grad`, both NumPy arrays. `float(loss.data)` gives the loss as a plain number.
:::

::: concept
### The same model with a quarter of the code

| step | Stage 1 (by hand) | engine |
| --- | --- | --- |
| look up | `embedding_forward` + `embedding_backward` | `C[x]` |
| flatten | `flatten_context` + reshape back | `ops.reshape(emb, (B, -1))` |
| linear | `linear_forward` + `linear_backward` | `flat @ W1 + b1` |
| tanh | `tanh_forward` + `tanh_backward` | `ops.tanh(...)` |
| output | `linear_forward` + `linear_backward` | `h @ W2 + b2` |
| loss | `cross_entropy` + `cross_entropy_backward` | `ops.cross_entropy(logits, y)` |
| backward | calls in hand-chosen order in `backward(cache)` | `loss.backward()` |

No caches, no reverse order. `unbroadcast` handles the bias, and `getitem`'s `np.add.at` handles repeated ids in `C[x]`.

### Why the gradients must match exactly

Both versions compute the same function with the same formulas: your engine's operations are the Stage 1 derivations moved into small functions. With equal parameters and inputs, the loss and the gradients agree up to float32 rounding (the check allows a relative difference of $10^{-5}$). If they disagree, one of the two is wrong. Two independent implementations that must agree make an excellent test.

### A training step on the engine

Each step:

1. call `zero_grad()` on every parameter Tensor;
2. `loss = model(x, y)`;
3. `loss.backward()`;
4. for every parameter, `data -= lr * grad`.

The `zero_grad` step is new. The hand-written backward returned fresh arrays every step, but the engine adds into the leaves' gradients.

### What this gives Stage 2

A transformer block has layer normalization, attention with reshapes and transposes, residual additions and an MLP. Writing its backward pass by hand would be long and fragile. On the engine, a block is about 10 lines of forward code. Only truly new operations (LayerNorm, GELU, softmax) need a derived backward, and each is checked once on its own.
:::

::: think
Q: The engine's gradient for `b1` matches Stage 1, but the gradient for `C` does not. Which operation do you suspect first?
A: `getitem`, the lookup `C[x]`. Its backward must use `np.add.at` so that repeated ids add up. A `grad[key] += g` version agrees whenever the ids in the batch happen to be all different, and fails otherwise.
:::

::: think
Q: Why compare with your own Stage 1 code instead of numerical gradients?
A: Both work. Stage 1 was already checked with numerical gradients, so it tests the whole model at once, in float32, without float64 copies or two forward passes per entry.
:::

::: hint level=1
`__init__` stores the five arrays as Tensors. `__call__` is the s1-l09 forward pass written with Tensor operations. `parameters` returns the five Tensors by name.
:::

::: hint level=2
`Tensor(params["C"])` and so on. `self.C[x]` is the lookup, `ops.reshape(emb, (B, -1))` flattens, `@` and `+` build the linear layers, and `ops.tanh` and `ops.cross_entropy` finish it.
:::

::: hint level=3
```pseudo
__init__(params):  C, W1, b1, W2, b2 ← a Tensor of each array
parameters():      return {name: tensor} for the five
__call__(x, y):
    B ← number of rows of x
    emb  ← C[x]                         # (B, k, e)
    flat ← reshape(emb, (B, −1))        # (B, k·e)
    h    ← tanh(flat @ W1 + b1)         # (B, h)
    logits ← h @ W2 + b2                # (B, V)
    return cross_entropy(logits, y)
```
:::

::: attempt
File: `workspace/autograd/mlp.py` — fill in the `MLP` class.
Check: `python -m course check a-l04`
:::

::: solution
@include solutions/autograd/mlp.py#MLP

- The forward pass reads like the table in s1-l09; the engine records the graph while it runs.
- `parameters()` returns the Tensors themselves, not copies, so changing their `.data` in place trains the model.
- There is no backward code at all: `loss.backward()` walks the recorded graph.
:::

::: breakit
Train the engine MLP for 50 steps **without** `zero_grad`, and print the size of `W1.grad` (`np.linalg.norm(W1.grad)`) at each step. It grows roughly in proportion to the step count, because each step adds a new gradient to all the old ones. The loss stops improving or blows up.
:::

::: recap
- The whole Stage 1 model is a dozen lines of forward code on your engine.
- Its loss and all gradients equal the hand-written version: two independent implementations agree.
- Training on the engine is: zero the gradients, forward, backward, update.
:::
