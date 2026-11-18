---
id: a-l05
title: Making the engine fast
stage: a
order: 5
prereqs: [a-l04]
files: [workspace/autograd/ops.py, workspace/autograd/tensor.py]
functions: [shared_weight_grad]
tests: tests/a/test_l05_speed.py
est_minutes: 45
---

::: goal
Your engine is correct; now you make it faster, the way engineers do it: measure first, then fix what the measurement points at. You will profile one training step, then make two changes. Gradients get allocated with `np.zeros`. The gradient of a shared weight becomes one large matrix product, which you derive from $\sum_b x_b^\top g_b = X^\top G$.
:::

::: notation
| Symbol | Meaning |
| --- | --- |
| $W$ | a weight matrix of shape $(n, p)$, shared by every row of the batch |
| $x_b$ | the input rows of batch entry $b$, shape $(T, n)$ |
| $g_b$ | the gradient arriving at the output rows of entry $b$, shape $(T, p)$ |
| $X$ | all input rows stacked into one matrix, shape $(BT, n)$ |
| $G$ | all output gradients stacked, shape $(BT, p)$ |
| $x_r, g_r$ | one row of $X$ and of $G$ (a row vector) |
| $B, T$ | batch size and sequence length |
| $r$ | a row of the stacked matrices, $r = 1 \dots BT$ |
| $t$ | a position inside one batch entry, $t = 1 \dots T$ |
| $x_{b,t}, g_{b,t}$ | row $t$ of $x_b$ and of $g_b$ |
| $dW$ | the gradient of the loss with respect to $W$ |
:::

::: numpy
### Finding the slow part: cProfile

Python's built-in profiler records how long every function takes:

```python
import cProfile, pstats
cProfile.run("step()", "prof.out")                  # run one training step under the profiler
pstats.Stats("prof.out").sort_stats("tottime").print_stats(8)   # the 8 most expensive functions
```

The `tottime` column is the time spent inside each function itself; `cumtime` also counts the functions it calls.

### Two ways to make an array of zeros

```python
x = np.zeros((32, 4, 128, 128), dtype=np.float32)   # 8 MB
np.zeros_like(x)                  # writes all 8 MB of zeros:       about 4.7 ms
np.zeros(x.shape, dtype=x.dtype)  # asks the OS for zeroed memory:  about 0.08 ms
```

Both return the same zeros. `np.zeros` gets memory that the operating system hands out already zeroed, page by page, only when it is first used. `np.zeros_like` writes every byte itself, right away.

### Stacking all rows

`reshape(-1, n)` turns any `(..., n)` array into one matrix with all rows stacked:

```python
a = np.zeros((16, 256, 256))
print(a.reshape(-1, 256).shape)   # (4096, 256)
```
:::

::: concept
### Measure first

Guessing where a program is slow is usually wrong. So measure: run one training step of the Stage 2 model under `cProfile`. On the reference machine, the profile of a 1.1-second step showed:

- `np.zeros_like` was called 283 times and took 0.21 s, about 20% of the step. Every Tensor creates a zero gradient when it is made, and `backward` resets every intermediate gradient again.
- matmul's backward was the largest single operation, as expected for a model made of matrix products.
- (The profile also led to a bug fix: GELU's constant `np.sqrt(2/π)` turned float32 into float64, s2-l03.)

### Fix 1: cheaper zero arrays

Change `np.zeros_like(arr)` to `np.zeros(arr.shape, dtype=arr.dtype)` in `Tensor.__init__` and in `backward`'s reset of intermediate gradients. The result is the same array of zeros, but creating it costs almost nothing (see the box). In the Stage 2 model this cut a step from 1.11 s to 0.95 s, about 15%.

### Fix 2: the gradient of a shared weight

A linear layer computes `x @ W` with `x` of shape `(B, T, n)` and `W` of shape `(n, p)`. In a-l02, its weight gradient was computed as $B$ separate products, followed by a sum over the batch:

$$
dW = \sum_{b=1}^{B} x_b^\top g_b
$$

`np.swapaxes(a, -1, -2) @ g` computes all $B$ products into a `(B, n, p)` array, and `unbroadcast` then sums it to `(n, p)`.

Now look inside one product. A matrix product is a sum of outer products of rows: $x_b^\top g_b = \sum_{t=1}^{T} x_{b,t}^\top g_{b,t}$, where $x_{b,t}$ is row $t$ of $x_b$. So

$$
dW = \sum_{b=1}^{B} \sum_{t=1}^{T} x_{b,t}^\top g_{b,t} = \sum_{r=1}^{BT} x_r^\top g_r = X^\top G
$$

The last step stacks all $BT$ rows into $X$ `(BT, n)` and $G$ `(BT, p)`: the double sum over batch entries and positions is one sum over all rows, and that is exactly one matrix product. In code: `a.reshape(-1, n).T @ g.reshape(-1, p)`.

What this gains, measured on one CPU thread:

| case | $B$ products + sum | one stacked product | temporary array |
| --- | --- | --- | --- |
| Stage 2 `qkv`, `(32, 128, 128) @ (128, 384)` | 4.2 ms | 3.1 ms | |
| Stage 3 output layer, `(16, 256, 256) @ (256, 4096)` | 84 ms | 65 ms | 67 MB → none |

About 1.3 times faster, and the `(B, n, p)` temporary array disappears: for the Stage 3 output layer that is $16 \times 256 \times 4096$ numbers, 67 MB, created and thrown away at every step. One large product also suits the linear-algebra library better than many small ones, especially with several threads.

The two methods give the same numbers, up to float32 rounding (about $10^{-4}$ here, from adding the terms in a different order).

### Where to use it

In `matmul`'s backward: when `b` is a 2-D weight and `a` has batch axes (`a.data.ndim > 2`), compute `b.grad += shared_weight_grad(a.data, g)`. Keep the old line for every other case: batched products such as $QK^\top$, where both inputs have batch axes and nothing is shared.

### Memory: the third finding

The profile can't show it, but the operating system's process monitor does: memory grew by hundreds of MB per step. Every Tensor's backward function refers to the Tensor itself, a "reference cycle", which Python frees only now and then. s2-l09 handles it with `gc.collect()` after each step.
:::

::: think
Q: Why is $\sum_b x_b^\top g_b$ equal to $X^\top G$? Explain with the entry $(i, j)$.
A: Entry $(i, j)$ of $x_b^\top g_b$ is $\sum_t x_{b,t,i}\, g_{b,t,j}$. Adding over $b$ gives $\sum_b \sum_t x_{b,t,i}\, g_{b,t,j}$: a sum over every row of every batch entry. Entry $(i, j)$ of $X^\top G$ is $\sum_r X_{r,i} G_{r,j}$, the same sum over the same rows, only numbered $r = 1 \dots BT$.
:::

::: think
Q: Why may you not use the stacked product for $Q K^\top$ in attention?
A: There, both inputs have batch axes: sequence $b$'s queries are multiplied only with sequence $b$'s keys. No weight is shared across the batch, so the gradient of each $K_b$ stays separate; summing over the batch would mix the sequences.
:::

::: think
Q: The profile showed `np.zeros_like` taking 20% of the step. Why doesn't Fix 1 save the full 20%?
A: `np.zeros` defers the work: the operating system zeroes each page of memory the first time it is written. Gradients that the backward pass fills in still pay that cost then. Only the zeroing of arrays that are never fully used is saved for good, so the measured gain was about 15%.
:::

::: hint level=1
`shared_weight_grad` is one line: stack the rows of `a` and of `g`, then one matrix product. In `matmul`, call it only in the "2-D weight, batched input" case. Fix 1 is a one-word change in two places in `tensor.py`.
:::

::: hint level=2
`n, p = a.shape[-1], g.shape[-1]`, then `a.reshape(-1, n).T @ g.reshape(-1, p)`. In `matmul`'s backward: `if b.data.ndim == 2 and a.data.ndim > 2: b.grad += shared_weight_grad(a.data, g)`, else the old `unbroadcast(...)` line. In `tensor.py`: `np.zeros(arr.shape, dtype=arr.dtype)`.
:::

::: hint level=3
```pseudo
shared_weight_grad(a, g):
    n ← last size of a;  p ← last size of g
    return (a with all rows stacked, transposed) @ (g with all rows stacked)    # (n, p)

matmul backward:
    a.grad += unbroadcast(g @ b transposed, a's shape)      # unchanged
    if b is 2-D and a has batch axes: b.grad += shared_weight_grad(a.data, g)
    otherwise:                        b.grad += unbroadcast(a transposed @ g, b's shape)
```
:::

::: attempt
The editor below has a tab for each file. In `workspace/autograd/ops.py`: fill in `shared_weight_grad` and call it from your `matmul`; `workspace/autograd/tensor.py` — switch your gradient allocations to `np.zeros(shape, dtype=...)`.
Check: press **Run check** in the Check section below.
:::

::: solution
@include solutions/autograd/ops.py#shared_weight_grad

- `reshape(-1, n)` stacks every row, whatever the number of leading axes.
- `(n, BT) @ (BT, p)` is one large product: no `(B, n, p)` temporary array, no separate sum.

@include solutions/autograd/ops.py#matmul

- Only the shared-weight case changes; batched products keep the general formula.
:::

::: breakit
Profile one Stage 2 training step before and after your two changes (the box shows how). Compare the total time and the `tottime` of `zeros_like` and of matmul's backward. Then put a batched `QK^T` product through `shared_weight_grad` by mistake (drop the `ndim == 2` test): the a-l02 gradient checks for 4-D batched matmul fail, because sequences get mixed.
:::

::: recap
- Measure before you optimize: `cProfile` shows where a step spends its time.
- `np.zeros(shape, dtype)` gets lazily zeroed memory; `np.zeros_like` writes every byte.
- A weight shared by all rows has the gradient $\sum_b x_b^\top g_b = X^\top G$: one stacked product, no large temporary array.
:::
