---
id: s2-l05
title: Causal mask
stage: s2
order: 5
prereqs: [s2-l04]
files: [workspace/stage2/attention.py]
functions: [causal_mask]
tests: tests/s2/test_l05_causal.py
est_minutes: 35
---

::: goal
Stop each position from looking at the future: add a `(T, T)` mask of $0$ (allowed) and $-10^9$ (forbidden, key after query) to the scores before the softmax, so position $t$ attends only to positions $0 \dots t$.
:::

::: notation
| Symbol | Meaning |
| --- | --- |
| $T$ | sequence length |
| $i$ | query position (row of the scores) |
| $j$ | key position (column) |
| $M$ | the mask, $(T, T)$: $M_{ij} = 0$ if $j \le i$, else a large negative number |
| $S$ | attention scores $(B, T, T)$ |
| $P$ | attention weights |
| $e$ | Euler's number |
| $s$ | a score value |
:::

::: concept
### Why

During training, position $i$ predicts $y_i = x_{i+1}$ (s2-l01). If it could attend to position $i + 1$, it would simply copy the answer: training loss near 0, and a model that is useless at generation time, when the future doesn't exist yet. So position $i$ may only use positions $j \le i$.

### How: add before the softmax

$$
P = \mathrm{softmax}(S + M), \qquad M_{ij} = \begin{cases} 0 & j \le i \\ -10^9 & j > i \end{cases}
$$

For $T = 4$:

$$
M = \begin{bmatrix} 0 & -10^9 & -10^9 & -10^9 \\ 0 & 0 & -10^9 & -10^9 \\ 0 & 0 & 0 & -10^9 \\ 0 & 0 & 0 & 0 \end{bmatrix}
$$

After the max-subtraction in the softmax, a masked entry is about $-10^9$, and $e^{-10^9}$ is exactly 0 in float32. So masked weights are exactly zero, every row still sums to 1 over the allowed keys, and row 0 puts all its weight on position 0 itself. The mask is `(T, T)` and broadcasts over the batch (and, in s2-l06, the heads). It contains no parameters, so it is a plain NumPy array, added like a constant.

The gradient takes care of itself: a weight that is exactly 0 contributes $P_{ij}(\dots) = 0$ to $dS$, so no gradient flows to the future, and the mask itself gets none.

### Why $-10^9$ and not $-\infty$

$-\infty$ looks cleaner but invites `nan`:

- A fully masked row (every key forbidden, which happens with padding schemes) has max $-\infty$, and the max-subtraction computes $-\infty - (-\infty)$, which is `nan`.
- $0 \cdot \infty$ anywhere in a backward pass is `nan`.

$-10^9$ is finite, fits easily in float32 (max about $3.4 \times 10^{38}$), and is still so negative that its exponential is exactly 0. A fully masked row then becomes uniform instead of `nan`: wrong, but contained.

### Testing it: the perturbation test

The cleanest check of causality is behavioral: change the input at positions $> t$, and the outputs at positions $\le t$ must not change at all. The check does exactly this, for every $t$.
:::

::: think
Q: With the mask in place, what is row 0 of $P$, and what does the output at position 0 equal?
A: $[1, 0, 0, \dots]$: position 0 may only see itself. The output is $v_0$, its own value vector.
:::

::: think
Q: The mask is added to the scores. Why not multiply the weights $P$ by a 0/1 mask after the softmax instead?
A: Rows would no longer sum to 1 (the probability mass on future keys just disappears), so the outputs shrink by different amounts at different positions. Renormalizing would fix that, but adding $-10^9$ before the softmax does the masking and the renormalization in one step.
:::

::: think
Q: During generation we feed the whole context and only use the last position's prediction. Why is the mask still needed there?
A: Strictly it isn't for the last position (it may see everything before it). But the model was trained with the mask, so every earlier position's representation, which later layers use, must be computed the same way as in training. Without the mask the hidden states differ from anything seen in training.
:::

::: hint level=1
Build a $(T, T)$ array that is $-10^9$ strictly above the diagonal and 0 elsewhere; in `attention_single_head`, add it to the scores when `causal` is true.
:::

::: hint level=2
`np.triu(array, k=1)` keeps the part strictly above the diagonal and zeroes the rest; start from `np.full((T, T), NEG, dtype=np.float32)`. Adding a NumPy array to a Tensor works (the engine wraps it as a constant).
:::

::: hint level=3
```pseudo
causal_mask(T):
    return the strictly-upper-triangular part of a (T, T) array full of NEG   # float32

attention_single_head(..., causal):
    ...
    if causal: scores ← scores + causal_mask(T)
    ...
```
:::

::: attempt
File: `workspace/stage2/attention.py` — implement `causal_mask` and use it in `attention_single_head` when `causal=True`.
Check: `python -m course check s2-l05`
:::

::: solution
@include solutions/stage2/attention.py#causal_mask

- `np.triu(..., k=1)` keeps only $j > i$; `NEG = -1e9` is finite, so no `inf - inf`.

@include solutions/stage2/attention.py#attention_single_head

- The mask is added before the softmax; the `(T, T)` array broadcasts over the batch.
:::

::: breakit
Train the full model (s2-l09) for 300 steps with `causal=False` everywhere. Expect the training loss to fall far below anything the causal model reaches (toward 0.1), because each position can read its own target; then sample from it and expect garbage, since at generation time the "future" it learned to copy isn't there.
:::

::: recap
- Position $i$ may attend only to $j \le i$; otherwise training leaks the answer.
- Add $0 / -10^9$ to the scores before the softmax: masked weights become exactly 0 and rows still sum to 1.
- Use a large finite negative number, not $-\infty$, to avoid `nan`.
- The perturbation test checks causality directly.
:::
