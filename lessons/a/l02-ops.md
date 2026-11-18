---
id: a-l02
title: Broadcasting-aware ops
stage: a
order: 2
prereqs: [a-l01]
files: [workspace/autograd/tensor.py, workspace/autograd/ops.py]
functions: [unbroadcast, matmul, tanh, sum, mean, reshape, transpose, getitem, cross_entropy]
tests: tests/a/test_l02_ops.py
est_minutes: 75
---

::: goal
You will give the engine every operation the MLP and the transformer need, each with its own backward step: `matmul`, `tanh`, `sum`, `mean`, `reshape`, `transpose`, `getitem` (for lookups) and a combined `cross_entropy`. You also write `unbroadcast`, so that `+` and `*` work on shapes like `(B, m) + (m,)`.
:::

::: notation
| Symbol | Meaning |
| --- | --- |
| $a, b$ | input tensors of an operation |
| $c$ | an output tensor |
| $g$ | the gradient arriving from above, `out.grad`, with the output's shape |
| $B, m, n, p$ | sizes: batch, and matrix sides |
| $\top$ | transpose of the last two axes |
| $\pi$ | a new order of the axes (for `transpose`) |
| $\pi^{-1}$ | the order that undoes $\pi$ |
| $N$ | number of rows (for cross-entropy) |
| $V$ | number of characters |
| $y$ | the correct ids |
| $p$ | softmax probabilities |
| $j$ | an output column |
| $b_j$ | entry $j$ of a bias vector $b$ |
| $c_{rj}$ | entry of the output in row $r$, column $j$ |
| $r$ | a row |
| $L$ | the loss at the end of the graph |
:::

::: numpy
### Swap the last two axes (n-l02)

```python
x = np.zeros((2, 3, 4))
print(np.swapaxes(x, -1, -2).shape)     # (2, 4, 3)
```

### Put a removed axis back, then stretch it

`np.expand_dims(s, axis)` inserts an axis of length 1. `np.broadcast_to(s, shape)` stretches length-1 axes to a larger shape, without copying:

```python
s = np.array([3., 7.])                   # row sums of a (2, 3) array, shape (2,)
print(np.expand_dims(s, 1).shape)        # (2, 1)
print(np.broadcast_to(np.expand_dims(s, 1), (2, 3)))
# [[3. 3. 3.]
#  [7. 7. 7.]]
```

### Undo a transpose

`np.argsort(order)` gives the axis order that undoes `order`:

```python
order = (2, 0, 1)
print(np.argsort(order))                 # [1 2 0]
y = np.transpose(np.zeros((2, 3, 4)), order)
print(y.shape)                           # (4, 2, 3)
print(np.transpose(y, np.argsort(order)).shape)   # (2, 3, 4)   back to the start
```

### Sum and keep the axis (n-l04)

```python
g = np.ones((4, 3))
print(g.sum(axis=0).shape, g.sum(axis=0, keepdims=True).shape)   # (3,) (1, 3)
```
:::

::: concept
### Unbroadcasting

In `c = a + b`, with `a` of shape `(2, 3)` and `b` of shape `(3,)`, NumPy treats `b` as `(1, 3)` and stretches it over both rows:

$$
c_{rj} = a_{rj} + b_j, \qquad r \in \{0, 1\}, \ j \in \{0, 1, 2\}
$$

$b_j$ appears in **both** rows of $c$, so by the rule "add up all paths":

$$
\frac{\partial L}{\partial b_j} = \sum_{r} g_{rj} \cdot 1 = g_{0j} + g_{1j}
$$

Example: $g = \begin{bmatrix} 0 & 1 & 2 \\ 3 & 4 & 5 \end{bmatrix}$ gives $\frac{\partial L}{\partial b} = [3, 5, 7]$. This is `db = dout.sum(axis=0)` from s1-l06, in general form. The rule: **every axis that broadcasting stretched in the forward pass gets summed in the backward pass.** There are two cases:

1. **Added axes.** NumPy lines shapes up from the right and adds missing axes at the front: `(3,)` against `(2, 3)` acts as `(1, 3)`. Sum the gradient over the extra front axes until it has as many axes as the input.
2. **Stretched length-1 axes.** If the input has length 1 on an axis where the output is longer, sum over that axis with `keepdims=True`, so the length-1 axis stays. For `(2, 1)` against `(2, 3)`: sum over axis 1, which gives `(2, 1)`.

`unbroadcast(grad, shape)` does both, and `+` and `*` call it for each input. In `*`, multiply by the local derivative first: `unbroadcast(g * other.data, self.shape)`.

### matmul

From s1-l06 and the shape-matching trick: for $c = a\,b$ with $a$ `(n, m)` and $b$ `(m, p)`,

$$
\frac{\partial L}{\partial a} = g\, b^\top \quad (n, p)(p, m) = (n, m), \qquad
\frac{\partial L}{\partial b} = a^\top g \quad (m, n)(n, p) = (m, p)
$$

The transformer uses batched products like `(B, n, m) @ (m, p)`, which apply the same formula to each batch entry. So transpose only the **last two** axes (`np.swapaxes(x, -1, -2)`). The weight `(m, p)` is shared by all $B$ entries, so its gradient `a^T g` comes out as `(B, m, p)` and must be summed over $B$. `unbroadcast` does that too.

### tanh, reshape, transpose

- tanh: $g \cdot (1 - t^2)$, where $t$ is the output (s1-l06).
- reshape only relabels positions, so its backward reshapes $g$ back to the input's shape (s1-l07).
- transpose with axis order $\pi$ puts input axis $\pi_i$ at position $i$. Its backward puts them back with the order that undoes it, $\pi^{-1}$ = `np.argsort(pi)`. Example: $\pi = (2, 0, 1)$ turns `(2, 3, 4)` into `(4, 2, 3)`; $\pi^{-1} = (1, 2, 0)$ turns `(4, 2, 3)` back into `(2, 3, 4)`.

### sum and mean

`c = a.sum(axis=1)` with $a$ `(2, 3)` means $c_r = a_{r0} + a_{r1} + a_{r2}$, so every $a_{rj}$ receives $g_r \cdot 1$. The backward copies $g$ back across the summed axis: put the axis back (if `keepdims` was False) and stretch to `a`'s shape. `mean` is `sum` times $\frac{1}{\text{count}}$. Build it from `sum` and `*`, and the engine handles the rest.

### getitem: lookups and slices

`C[ids]` is the lookup of s1-l02, and `x[:, 1:3]` is a slice; both are `getitem`. The backward sends $g$ back to the positions the entries came from, **adding** repeated positions: `np.add.at(a.grad, key, g)`, as in s1-l07. Slices have no repeated positions, and `np.add.at` still works for them.

### A combined cross-entropy

We could build cross-entropy from `exp`, `sum`, `log` and indexing. That would create several large `(N, V)` arrays and lose the overflow protection of s1-l04. Instead it is one operation. The forward pass is your safe log-sum-exp. The backward pass is the result of s1-l05, $\frac{1}{N}(p - \text{onehot}(y))$, multiplied by `out.grad` (which is 1 when the loss is the end of the graph). One operation, one derived formula, no overflow.
:::

::: think
Q: `a` has shape `(4, 1, 3)` and `b` has shape `(2, 1)`. What is the shape of `a * b`, and what does `unbroadcast` do to the gradient for `b`?
A: Lined up from the right, `b` acts as `(1, 2, 1)`, so the result is `(4, 2, 3)`. For `b`: sum away the front axis (giving `(2, 3)`), then sum axis 1, where `b` had length 1, keeping it (giving `(2, 1)`).
:::

::: think
Q: In `h @ W`, with $h$ `(B, T, d)` and $W$ `(d, d)`, why must the gradient of $W$ be summed over the batch, and which part of your code does it?
A: The same $W$ was used for each of the $B$ batch entries, so its gradient collects $B$ contributions. `swapaxes(h) @ g` gives `(B, d, d)`, and `unbroadcast` to `(d, d)` sums over $B$.
:::

::: think
Q: Why give cross-entropy its own combined backward instead of building it from smaller operations?
A: Safety and cost. The built-up version would compute `exp` of raw scores (overflow), `log` of tiny probabilities ($-\infty$), and store several `(N, V)` arrays. The combined version uses log-sum-exp going forward and the clean $p - \text{onehot}$ formula going backward.
:::

::: hint level=1
Every operation has the same pattern: compute `out_data` with NumPy; create `out = Tensor(out_data, (inputs...), "name")`; define a `_backward` that does `input.grad += <local derivative applied to out.grad>`; store it on `out`; return `out`. Only `unbroadcast` is not an operation.
:::

::: hint level=2
`unbroadcast`: a `while` loop that sums `axis=0` until `grad.ndim == len(shape)`, then a loop over the axes where `shape` has length 1. `matmul`: `np.swapaxes(x, -1, -2)`, and `unbroadcast` on both results. `sum`: `np.expand_dims` (when `keepdims` is False), then `np.broadcast_to`. `transpose`: `np.argsort(axes)`. `getitem`: `np.add.at`. `cross_entropy`: reuse your s1-l04 and s1-l05 code, and multiply the gradient by `out.grad`.
:::

::: hint level=3
```pseudo
unbroadcast(grad, shape):
    while grad has more axes than shape: grad ← grad summed over axis 0
    for each axis where shape[axis] = 1 and grad's length there ≠ 1:
        grad ← grad summed over that axis, keeping it
    return grad

matmul(a, b):
    out ← Tensor(a.data @ b.data, (a, b))
    backward: a.grad += unbroadcast(g @ (b.data with last two axes swapped), a's shape)
              b.grad += unbroadcast((a.data with last two axes swapped) @ g, b's shape)

sum(a, axis, keepdims):
    backward: g' ← g with the summed axes put back (if not keepdims)
              a.grad += g' stretched to a's shape
```
:::

::: attempt
Files: `workspace/autograd/tensor.py` — fill in `unbroadcast` and use it in `__add__` and `__mul__`; `workspace/autograd/ops.py` — fill in every operation.
Check: `python -m course check a-l02`
:::

::: solution
@include solutions/autograd/tensor.py#unbroadcast

- The `while` loop handles added front axes; the `for` loop handles stretched length-1 axes and keeps them, so the result has exactly `shape`.

@include solutions/autograd/ops.py#matmul

- Swapping only the last two axes makes the same code work for 2-D and batched products. `unbroadcast` sums the gradient of a shared weight over the batch axes.

@include solutions/autograd/ops.py#sum

@include solutions/autograd/ops.py#mean

- `mean` is built from `sum` and `*`, so it needs no backward of its own.

@include solutions/autograd/ops.py#transpose

@include solutions/autograd/ops.py#getitem

- `np.add.at` adds repeated positions, so this one operation is a correct lookup.

@include solutions/autograd/ops.py#cross_entropy

- The forward pass is the log-sum-exp of s1-l04; the backward pass is the formula of s1-l05, scaled by `out.grad`, so the operation also works when the loss is not the end of the graph.

`tanh` and `reshape` follow the same pattern; see `solutions/autograd/ops.py`.
:::

::: breakit
In `__add__`, replace `unbroadcast(out.grad, other.data.shape)` with plain `out.grad`. Compute `y = x + b` with `x` `(4, 3)` and `b` `(3,)`, and run a backward pass. NumPy stops with `ValueError: non-broadcastable output operand with shape (3,) doesn't match the broadcast shape (4, 3)` at `other.grad += ...`: the bias gradient has the wrong shape, exactly as in the break-it of s1-l06.
:::

::: recap
- Broadcasting in the forward pass means summing in the backward pass; `unbroadcast` does it for every operation.
- matmul's backward is `g bᵀ` and `aᵀ g` on the last two axes, plus `unbroadcast` for shared weights.
- `getitem` with `np.add.at` handles lookups and slices; cross-entropy is one combined, safe operation.
:::
