---
id: s2-l06
title: Multi-head attention
stage: s2
order: 6
prereqs: [s2-l05]
files: [workspace/stage2/attention.py]
functions: [MultiHeadAttention]
tests: tests/s2/test_l06_mha.py
est_minutes: 60
---

::: goal
You will run $h$ attention heads of size $d_h = d / h$ side by side. One combined projection computes all queries, keys and values at once; reshapes and transposes turn `(B, T, d)` into `(B, h, T, d_h)` and back; and an output projection mixes the heads.
:::

::: notation
| Symbol | Meaning |
| --- | --- |
| $h$ | number of heads (`n_heads`), 4 here |
| $d$ | model width, 128 |
| $d_h$ | head size $d / h$, 32 |
| $B, T$ | batch size, sequence length |
| $W_{qkv}$ | the combined projection, $(d, 3d)$, with a bias |
| $W_o$ | the output projection, $(d, d)$ (`proj`), with a bias |
| $Q, K, V$ | queries, keys and values of every head, $(B, h, T, d_h)$ |
:::

::: numpy
### Reshape, then transpose (n-l02)

A reshape only relabels the numbers in their current order; a transpose changes which axis is which. Splitting heads needs both, in this order:

```python
t = np.zeros((2, 5, 3 * 4 * 8))                          # (B=2, T=5, 3d) with h=4, d_h=8
print(t.reshape(2, 5, 3, 4, 8).transpose(2, 0, 3, 1, 4).shape)   # (3, 2, 4, 5, 8)
```

The reshape reads the last axis of length $3d$ as $3 \times h \times d_h$. The transpose `(2, 0, 3, 1, 4)` puts old axis 2 (q/k/v) first, then the batch, the heads, the positions and the head entries.

### Matrix products over two batch axes

`@` treats all axes except the last two as batch axes: `(B, h, T, d_h) @ (B, h, d_h, T)` gives `(B, h, T, T)`, one table per sequence and per head.
:::

::: concept
### Why several heads

One head computes one set of attention weights per position, so it can look for one kind of thing at a time. Several heads, each with its own projections, can look for different things: one head can follow the previous character, another the start of the current word, a third a matching quote mark. Each head works in a smaller space ($d_h = 32$ instead of 128), so the total cost is about the same as one big head.

### One combined projection

Instead of $3h$ small matrices, one $W_{qkv}$ of shape $(d, 3d)$ computes everything in a single matrix product: `(B, T, d) @ (d, 3d) → (B, T, 3d)`. We fix the order of the $3d$ entries:

$$
[\ \underbrace{q_{\text{head }0} \mid q_{\text{head }1} \mid \dots}_{d}\ \mid\ \underbrace{k_{\text{head }0} \mid \dots}_{d}\ \mid\ \underbrace{v_{\text{head }0} \mid \dots}_{d}\ ]
$$

so the axis splits as $3 \times h \times d_h$.

### Reshape and transpose to put the heads in the batch

| step | operation | shape |
| --- | --- | --- |
| 1 | `qkv(x)` | $(B, T, 3d)$ |
| 2 | reshape | $(B, T, 3, h, d_h)$ |
| 3 | transpose `(2, 0, 3, 1, 4)` | $(3, B, h, T, d_h)$ |
| 4 | take `[0]`, `[1]`, `[2]` | $Q, K, V$: $(B, h, T, d_h)$ each |
| 5 | scores $QK^\top/\sqrt{d_h}$ (+ mask) | $(B, h, T, T)$ |
| 6 | softmax, then times $V$ | $(B, h, T, d_h)$ |
| 7 | transpose `(0, 2, 1, 3)` | $(B, T, h, d_h)$ |
| 8 | reshape | $(B, T, d)$: the heads side by side |
| 9 | `proj` | $(B, T, d)$ |

Step 2 only relabels: the last axis, of length $3d$, is read as $3 \times h \times d_h$, in exactly the order fixed above. Step 3 moves the q/k/v axis to the front and the head axis next to the batch. Steps 5 and 6 are then ordinary batched matrix products over the front axes $(B, h)$: the same code as for one head. Step 7 undoes the head move, and step 8 joins the $h$ outputs of size $d_h$ back into one vector of size $d$. `proj` lets the heads' results mix.

A transpose gives a non-contiguous view (s0-l00), so the reshape in step 8 has to copy. Your engine's `reshape` calls NumPy's, which does this automatically.

Your engine already has every operation needed: `reshape`, `transpose`, `getitem` for `[0]`, batched `matmul` (with $K^\top$ = `transpose(k, (0, 1, 3, 2))`), `softmax`, and the mask, which broadcasts from $(T, T)$ to $(B, h, T, T)$.

### How the check tests it

The check compares your module with a slow version that loops over the heads. It cuts columns $[i d_h, (i+1) d_h)$ out of $Q$, $K$ and $V$ for head $i$, and joins the results. If your reshape uses a different order from the one fixed above, the shapes still match, but the numbers do not.
:::

::: think
Q: With $d = 128$ and $h = 4$, how many parameters does one multi-head attention layer have? Compare with four separate heads of size 32 plus an output projection.
A: $W_{qkv}$: $128 \cdot 384 + 384 = 49{,}536$; `proj`: $128 \cdot 128 + 128 = 16{,}512$; together $66{,}048$. Four heads with $W_q, W_k, W_v$ of $128 \times 32$ each have $4 \cdot 3 \cdot 4{,}096 = 49{,}152$ weights, the same as $W_{qkv}$ without its bias. Combining the projections changes the speed, not the number of parameters.
:::

::: think
Q: Why transpose to $(3, B, h, T, d_h)$ instead of reshaping $(B, T, 3d)$ directly into $(3, B, h, T, d_h)$?
A: A reshape never moves numbers; it only reads them in their current order. In memory, the $3d$ entries of each position sit together, so a direct reshape to a shape that starts with 3 would mix up positions and q/k/v. Reshape along the existing order, to $(B, T, 3, h, d_h)$, and then transpose.
:::

::: hint level=1
Follow the 9-row table; every row is one engine operation. Store `qkv` and `proj` as `Linear` modules, so their parameters are found by name.
:::

::: hint level=2
`ops.reshape(t, (B, T, 3, h, dh))`, `ops.transpose(t, (2, 0, 3, 1, 4))`, then `t[0]`, `t[1]`, `t[2]` (getitem). For $K^\top$ in 4-D: `ops.transpose(k, (0, 1, 3, 2))`. Reuse your `softmax` and `causal_mask`.
:::

::: hint level=3
```pseudo
__call__(x, causal):
    B, T, d ← x's shape;  h ← n_heads;  dh ← d / h
    qkv ← transpose(reshape(qkv_linear(x), (B, T, 3, h, dh)), (2, 0, 3, 1, 4))
    q, k, v ← qkv[0], qkv[1], qkv[2]                         # (B, h, T, dh)
    scores ← q @ transpose(k, (0, 1, 3, 2)) · 1/sqrt(dh)     # (B, h, T, T)
    if causal: scores ← scores + causal_mask(T)
    y ← softmax(scores) @ v                                  # (B, h, T, dh)
    y ← reshape(transpose(y, (0, 2, 1, 3)), (B, T, d))
    return proj(y)
```
:::

::: attempt
File: `workspace/stage2/attention.py` — fill in `MultiHeadAttention`.
Check: `python -m course check s2-l06`
:::

::: solution
@include solutions/stage2/attention.py#MultiHeadAttention

- The heads live in a batch axis, so one set of batched matrix products computes all of them.
- `qkv[0]` and the others are `getitem` operations; their backward sends each part of the gradient back into the right slice of the combined projection's output.
- The `(T, T)` mask broadcasts over `(B, h)`.
:::

::: breakit
Change the reshape order to `(B, T, h, 3, dh)`, keeping everything else. The check fails on values but not on shapes. The model would still train (it would learn a different layout), which is why the layout must be fixed, and tested, whenever two implementations share weights or weights are saved to disk.
:::

::: recap
- $h$ heads of size $d/h$ attend separately; `proj` mixes their outputs.
- One combined $(d, 3d)$ projection, then reshape to `(B, T, 3, h, dh)` and transpose to `(3, B, h, T, dh)`.
- The heads travel as a batch axis through the same matrix products and softmax.
:::
