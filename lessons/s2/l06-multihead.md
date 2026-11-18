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
Run $h$ attention heads of size $d_h = d / h$ in parallel with one fused QKV projection, using reshapes and transposes, `(B, T, d)` → `(B, h, T, d_h)` and back, followed by an output projection.
:::

::: notation
| Symbol | Meaning |
| --- | --- |
| $h$ | number of heads (`n_heads`), 4 here |
| $d$ | model width, 128 |
| $d_h$ | head size $d / h$, 32 |
| $B, T$ | batch size, sequence length |
| $W_{qkv}$ | fused projection $(d, 3d)$, with bias |
| $W_o$ | output projection $(d, d)$ (`proj`), with bias |
| $Q, K, V$ | per-head queries, keys, values, $(B, h, T, d_h)$ |
:::

::: concept
### Why several heads

One head computes one set of attention weights per position: it can look for one kind of thing at a time. With several heads, each with its own projections, one head can track the previous character while another looks for the start of the current word and a third for a matching quote mark. Each head works in a smaller space ($d_h = 32$ instead of 128), so the total cost is about the same as one big head.

### One fused projection

Instead of $3h$ separate small matrices, one $W_{qkv}$ of shape $(d, 3d)$ computes everything in a single matmul: `(B, T, d) @ (d, 3d) → (B, T, 3d)`. We fix the layout of the $3d$ axis:

$$
[\ \underbrace{q_{\text{head }0} \mid q_{\text{head }1} \mid \dots}_{d}\ \mid\ \underbrace{k_{\text{head }0} \mid \dots}_{d}\ \mid\ \underbrace{v_{\text{head }0} \mid \dots}_{d}\ ]
$$

so it splits as $3 \times h \times d_h$.

### Reshape and transpose to put heads in the batch

| step | operation | shape |
| --- | --- | --- |
| 1 | `qkv(x)` | $(B, T, 3d)$ |
| 2 | reshape | $(B, T, 3, h, d_h)$ |
| 3 | transpose `(2, 0, 3, 1, 4)` | $(3, B, h, T, d_h)$ |
| 4 | index `[0]`, `[1]`, `[2]` | $Q, K, V$: $(B, h, T, d_h)$ each |
| 5 | scores $QK^\top/\sqrt{d_h}$ (+ mask) | $(B, h, T, T)$ |
| 6 | softmax, times $V$ | $(B, h, T, d_h)$ |
| 7 | transpose `(0, 2, 1, 3)` | $(B, T, h, d_h)$ |
| 8 | reshape | $(B, T, d)$: heads side by side |
| 9 | `proj` | $(B, T, d)$ |

Step 2 only relabels: the last axis of length $3d$ is read as $3 \times h \times d_h$ in exactly the layout above (C order: the rightmost index varies fastest). Step 3 moves the "which of q/k/v" axis to the front and the head axis next to the batch, so steps 5 and 6 are ordinary batched matmuls over the leading axes $(B, h)$, the same code as one head. Step 7 undoes the head move; step 8 glues the $h$ outputs of size $d_h$ back into one vector of size $d$. `proj` lets the heads' results mix.

A transpose produces a non-contiguous view (s0-l00), so the reshape in step 8 must copy. Your engine's `reshape` calls NumPy's, which does that automatically.

Your engine already has every op needed: `reshape`, `transpose`, `getitem` for `[0]`, batched `matmul` (with $K^\top$ = `transpose(k, (0, 1, 3, 2))`), `softmax` and the mask, which broadcasts from $(T, T)$ to $(B, h, T, T)$.

### Checking it

The check compares your module with a slow reference that loops over heads, slicing columns $[i d_h, (i+1) d_h)$ out of $Q$, $K$, $V$ and concatenating the results. If the layout of your reshape differs from the documented one, the numbers won't match even though the shapes do.
:::

::: think
Q: With $d = 128$ and $h = 4$, count the parameters of one multi-head attention layer, and compare with four independent single heads of size 32 plus an output projection.
A: $W_{qkv}$: $128 \cdot 384 + 384 = 49{,}536$; `proj`: $128 \cdot 128 + 128 = 16{,}512$; total $66{,}048$. Four heads with $W_q, W_k, W_v$ of $128 \times 32$ each have $4 \cdot 3 \cdot 4{,}096 = 49{,}152$ weights, the same as $W_{qkv}$ without its bias. Fusing changes the speed, not the parameter count.
:::

::: think
Q: Why transpose to $(3, B, h, T, d_h)$ instead of reshaping straight from $(B, T, 3d)$ to $(3, B, h, T, d_h)$?
A: A reshape never moves data; it only reinterprets the order of the numbers. In memory, the $3d$ entries of each position are contiguous, so a direct reshape to a shape with $3$ first would mix up positions and q/k/v. You must reshape along the existing order (to $(B, T, 3, h, d_h)$) and then transpose.
:::

::: hint level=1
Follow the 9-row table. Every row is one engine op. Store `qkv` and `proj` as `Linear` modules so their parameters are found by name.
:::

::: hint level=2
`ops.reshape(t, (B, T, 3, h, dh))`, `ops.transpose(t, (2, 0, 3, 1, 4))`, then `t[0]`, `t[1]`, `t[2]` (getitem). For $K^\top$ in 4-D: `ops.transpose(k, (0, 1, 3, 2))`. Reuse your `softmax` and `causal_mask`.
:::

::: hint level=3
```pseudo
__call__(x, causal):
    B, T, d ← x shape;  h ← n_heads;  dh ← d / h
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
File: `workspace/stage2/attention.py` — implement `MultiHeadAttention`.
Check: `python -m course check s2-l06`
:::

::: solution
@include solutions/stage2/attention.py#MultiHeadAttention

- The heads live in a batch axis, so one set of batched matmuls computes all of them.
- `qkv[0]` etc. are `getitem` ops; their backward scatters each head group's gradient back into the right slice of the fused projection's output.
- The mask `(T, T)` broadcasts over `(B, h)`.
:::

::: breakit
Swap the reshape order to `(B, T, h, 3, dh)` (keeping the rest). Expect the check to fail on values but not on shapes: the model would still train (it would just learn a different layout), which is why the layout must be fixed by convention, and tested, whenever weights are shared between implementations or saved to disk.
:::

::: recap
- $h$ heads of size $d/h$ attend independently; `proj` mixes their outputs.
- One fused $(d, 3d)$ projection, then reshape `(B, T, 3, h, dh)` and transpose `(3, B, h, T, dh)`.
- Heads ride along as a batch axis through the same matmuls and softmax.
:::
