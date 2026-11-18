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
Give the engine every operation the MLP and the transformer need, each with its own local backward: `matmul`, `tanh`, `sum`, `mean`, `reshape`, `transpose`, `getitem` (embedding lookups) and a fused `cross_entropy`, plus `unbroadcast`, so that `+` and `*` work on broadcast shapes like `(B, m) + (m,)`.
:::

::: notation
| Symbol | Meaning |
| --- | --- |
| $a, b$ | input tensors of an operation |
| $c$ | an output tensor |
| $g$ | the upstream gradient `out.grad`, shape of the output |
| $B, m, n, p$ | sizes: batch, and matrix dimensions |
| $\top$ | transpose of the last two axes |
| $\pi$ | a permutation of axes (for `transpose`) |
| $\pi^{-1}$ | the inverse permutation |
| $N$ | number of rows (for cross-entropy) |
| $V$ | number of classes |
| $y$ | integer targets |
| $p$ | softmax probabilities |
| $j$ | an output column index |
| $b_j$ | entry $j$ of a bias vector $b$ |
| $c_{rj}$ | entry of the output in row $r$, column $j$ |
| $r$ | a row index |
| $L$ | the loss at the root of the graph |
:::

::: concept
### Unbroadcasting

In `c = a + b` with `a` `(2, 3)` and `b` `(3,)`, NumPy treats `b` as `(1, 3)` and stretches it over both rows:

$$
c_{rj} = a_{rj} + b_j, \qquad r \in \{0, 1\}, \ j \in \{0, 1, 2\}
$$

$b_j$ appears in **both** rows of $c$, so by the sum-over-paths rule

$$
\frac{\partial L}{\partial b_j} = \sum_{r} g_{rj} \cdot 1 = g_{0j} + g_{1j}
$$

Example: $g = \begin{bmatrix} 0 & 1 & 2 \\ 3 & 4 & 5 \end{bmatrix}$ gives $\frac{\partial L}{\partial b} = [3, 5, 7]$. This is the `db = dout.sum(axis=0)` of s1-l06, now in general form. The rule: **whatever axis broadcasting stretched, the backward pass sums over.** Two cases:

1. **Added axes.** NumPy aligns shapes from the right and prepends missing axes: `(3,)` against `(2, 3)` acts as `(1, 3)`. Sum the gradient over those leading axes until it has as many axes as the operand.
2. **Stretched length-1 axes.** If the operand has length 1 on an axis where the output has more, sum over that axis with `keepdims=True` so the length-1 axis stays. `(2, 1)` against `(2, 3)`: sum over axis 1 → `(2, 1)`.

`unbroadcast(grad, shape)` does both, and `+` and `*` call it for each operand. In `*`, the local derivative comes first: `unbroadcast(g * other.data, self.shape)`.

### matmul

From s1-l06 with the shape-matching trick: for $c = a\,b$ with $a$ `(n, m)` and $b$ `(m, p)`,

$$
\frac{\partial L}{\partial a} = g\, b^\top \quad (n, p)(p, m) = (n, m), \qquad
\frac{\partial L}{\partial b} = a^\top g \quad (m, n)(n, p) = (m, p)
$$

Batched products `(B, n, m) @ (m, p)` (used everywhere in the transformer) apply the same formula to each batch entry: transpose only the **last two** axes (`np.swapaxes(x, -1, -2)`). The weight `(m, p)` was shared by all $B$ entries, so its gradient `a^T g` comes out `(B, m, p)` and must be summed over $B$: that is `unbroadcast` again.

### tanh, reshape, transpose

- tanh: $g \cdot (1 - t^2)$ with $t$ the output (s1-l06).
- reshape only relabels positions, so its backward reshapes $g$ back to the input's shape (s1-l07).
- transpose with axis order $\pi$ moves axis $\pi_i$ of the input to position $i$. Its backward moves them back with the **inverse** permutation $\pi^{-1}$ (`np.argsort(pi)`). Example: $\pi = (2, 0, 1)$ sends `(2, 3, 4)` to `(4, 2, 3)`; $\pi^{-1} = (1, 2, 0)$ sends `(4, 2, 3)` back to `(2, 3, 4)`.

### sum and mean

`c = a.sum(axis=1)` with $a$ `(2, 3)`: $c_r = a_{r0} + a_{r1} + a_{r2}$, so every $a_{rj}$ gets $g_r \cdot 1$. The backward copies $g$ back across the summed axis: re-insert the axis (if `keepdims` was False) and broadcast to `a`'s shape. `mean` is `sum` times $\frac{1}{\text{count}}$; build it from `sum` and `*` and the engine does the rest.

### getitem: lookups and slices

`C[ids]` is the embedding lookup of s1-l02, and `x[:, 1:3]` is a slice; both are `getitem`. The backward scatters $g$ back to where each element came from, **accumulating** repeated indices: `np.add.at(a.grad, key, g)`, the s1-l07 lesson. For plain slices there are no repeats, and `np.add.at` still works.

### A fused cross-entropy

We could build cross-entropy from `exp`, `sum`, `log` and indexing, but that creates several large `(N, V)` intermediates and loses the numerical care of s1-l04. Instead it is one op: the forward pass is your stable log-sum-exp; the backward pass is the derivation of s1-l05, $\frac{1}{N}(p - \text{onehot}(y))$, multiplied by the upstream scalar `out.grad` (which is 1 when the loss is the root). One op, one derived formula, no instability.
:::

::: think
Q: `a` has shape `(4, 1, 3)` and `b` has shape `(2, 1)`. What is the shape of `a * b`, and what does `unbroadcast` do to the gradient for `b`?
A: Aligning from the right, `b` acts as `(1, 2, 1)`, so the result is `(4, 2, 3)`. For `b`: sum away the leading axis (→ `(2, 3)`), then sum axis 1, where `b` had length 1, with keepdims (→ `(2, 1)`).
:::

::: think
Q: In `h @ W` with $h$ `(B, T, d)` and $W$ `(d, d)`, why must the gradient of $W$ be summed over the batch, and which part of your code does it?
A: The same $W$ was used for every one of the $B$ batch entries (it was broadcast across them), so its gradient collects $B$ contributions. `swapaxes(h) @ g` produces `(B, d, d)`; `unbroadcast` to `(d, d)` sums over $B$.
:::

::: think
Q: Why give cross-entropy a fused backward rather than composing it from smaller ops?
A: Stability and cost. The composed version would compute `exp` of raw logits (overflow), `log` of tiny probabilities (−inf), and store several `(N, V)` arrays. The fused version uses log-sum-exp in the forward pass and the clean $p - \text{onehot}$ formula in the backward.
:::

::: hint level=1
Every op has the same shape: compute `out_data` with NumPy, `out = Tensor(out_data, (inputs...), "name")`, define `_backward` that does `input.grad += <local derivative applied to out.grad>`, attach it, return `out`. Only `unbroadcast` is not an op.
:::

::: hint level=2
`unbroadcast`: a `while` loop summing `axis=0` until `grad.ndim == len(shape)`, then a loop over the axes where `shape` has 1. `matmul`: `np.swapaxes(x, -1, -2)`, plus `unbroadcast` on both results. `sum`: `np.expand_dims` (when not keepdims) then `np.broadcast_to`. `transpose`: `np.argsort(axes)`. `getitem`: `np.add.at`. `cross_entropy`: copy your s1-l04/l05 code, multiply the gradient by `out.grad`.
:::

::: hint level=3
```pseudo
unbroadcast(grad, shape):
    while grad has more axes than shape: grad ← sum of grad over axis 0
    for each axis where shape[axis] = 1 and grad's length there ≠ 1:
        grad ← sum over that axis, keeping it
    return grad

matmul(a, b):
    out ← Tensor(a.data @ b.data, (a, b))
    backward: a.grad += unbroadcast(g @ swap_last_two(b.data), a's shape)
              b.grad += unbroadcast(swap_last_two(a.data) @ g, b's shape)

sum(a, axis, keepdims):
    backward: g' ← g with the summed axes re-inserted (if not keepdims)
              a.grad += g' broadcast to a's shape
```
:::

::: attempt
Files: `workspace/autograd/tensor.py` — implement `unbroadcast` and use it in `__add__` and `__mul__`; `workspace/autograd/ops.py` — implement every op.
Check: `python -m course check a-l02`
:::

::: solution
@include solutions/autograd/tensor.py#unbroadcast

- The `while` handles prepended axes; the `for` handles stretched length-1 axes, keeping them so the result has exactly `shape`.

@include solutions/autograd/ops.py#matmul

- Transposing only the last two axes makes the same code serve 2-D and batched products; `unbroadcast` sums the gradient of a shared weight over the batch axes.

@include solutions/autograd/ops.py#sum

@include solutions/autograd/ops.py#mean

- `mean` is composed from `sum` and `*`; it needs no backward of its own.

@include solutions/autograd/ops.py#transpose

@include solutions/autograd/ops.py#getitem

- `np.add.at` makes repeated indices accumulate, so this one op is a correct embedding lookup.

@include solutions/autograd/ops.py#cross_entropy

- The forward pass is s1-l04's log-sum-exp; the backward is s1-l05's formula, scaled by `out.grad` so the op also works when the loss is not the root.

`tanh` and `reshape` follow the same pattern; see `solutions/autograd/ops.py`.
:::

::: breakit
In `__add__`, replace `unbroadcast(out.grad, other.data.shape)` with plain `out.grad`, then compute `y = x + b` with `x` `(4, 3)` and `b` `(3,)`, and run a backward pass. Expect `ValueError: non-broadcastable output operand with shape (3,) doesn't match the broadcast shape (4, 3)` at `other.grad += ...`: the bias gradient has the wrong shape, exactly as in s1-l06's break-it.
:::

::: recap
- Broadcasting in the forward pass means summing in the backward pass; `unbroadcast` does it for every op.
- matmul's backward is `g bᵀ` and `aᵀ g` on the last two axes, plus unbroadcast for shared weights.
- `getitem` with `np.add.at` gives embedding lookups and slices; cross-entropy is one fused, stable op.
:::
