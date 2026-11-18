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
You will stop each position from looking at later positions. Before the softmax, you add a `(T, T)` mask to the scores: $0$ where looking is allowed, and $-10^9$ where the key comes after the query. Then position $t$ only attends to positions $0 \dots t$.
:::

::: notation
| Symbol | Meaning |
| --- | --- |
| $T$ | sequence length |
| $i$ | query position (a row of the scores) |
| $j$ | key position (a column) |
| $M$ | the mask, $(T, T)$: $M_{ij} = 0$ if $j \le i$, otherwise a large negative number |
| $S$ | attention scores, $(B, T, T)$ |
| $P$ | attention weights |
| $e$ | Euler's number |
| $s$ | a score value |
:::

::: numpy
### Keep the part above the diagonal

`np.full(shape, value)` fills an array with one value. `np.triu(a, k=1)` keeps the entries strictly above the diagonal and sets the rest to 0:

```python
print(np.triu(np.full((4, 4), -1e9, dtype=np.float32), k=1))
# [[ 0.e+00 -1.e+09 -1.e+09 -1.e+09]
#  [ 0.e+00  0.e+00 -1.e+09 -1.e+09]
#  [ 0.e+00  0.e+00  0.e+00 -1.e+09]
#  [ 0.e+00  0.e+00  0.e+00  0.e+00]]
```

NumPy prints the numbers in scientific notation: `0.e+00` is 0 and `-1.e+09` is $-10^9$.

### Adding a plain array to a Tensor

`scores + mask` works when `scores` is a Tensor and `mask` is a NumPy array: the engine wraps the array as a constant. The `(T, T)` mask broadcasts over the batch.
:::

::: concept
### Why

During training, position $i$ predicts $y_i = x_{i+1}$ (s2-l01). If it could attend to position $i + 1$, it would just copy the answer. The training loss would fall close to 0, and the model would be useless when it writes text, because then the next character does not exist yet. So position $i$ may only use positions $j \le i$.

### How: add the mask before the softmax

$$
P = \mathrm{softmax}(S + M), \qquad M_{ij} = \begin{cases} 0 & j \le i \\ -10^9 & j > i \end{cases}
$$

For $T = 4$:

$$
M = \begin{bmatrix} 0 & -10^9 & -10^9 & -10^9 \\ 0 & 0 & -10^9 & -10^9 \\ 0 & 0 & 0 & -10^9 \\ 0 & 0 & 0 & 0 \end{bmatrix}
$$

After the softmax subtracts the maximum, a masked score is about $-10^9$, and $e^{-10^9}$ is exactly 0 in float32. So the masked weights are exactly zero, each row still adds up to 1 over the allowed keys, and row 0 puts all its weight on position 0 itself. The mask is `(T, T)` and broadcasts over the batch (and, in s2-l06, over the heads). It has no parameters, so it is a plain NumPy array, added like a constant.

The gradient needs no special care. A weight that is exactly 0 adds exactly 0 to $dS$, so no gradient flows to later positions, and the mask itself gets none.

### Why $-10^9$ and not $-\infty$

$-\infty$ looks cleaner, but it invites `nan`:

- A row where every key is masked (this happens with some padding schemes) has the maximum $-\infty$. Subtracting the maximum then computes $-\infty - (-\infty)$, which is `nan`.
- $0 \cdot \infty$ anywhere in a backward pass is `nan`.

$-10^9$ is finite, fits easily into float32 (whose largest value is about $3.4 \times 10^{38}$), and is still so negative that its exponential is exactly 0. A fully masked row then becomes uniform instead of `nan`: wrong, but harmless.

### Testing it: change the future, check the past

The cleanest test of the mask: change the input at positions $> t$, and check that the outputs at positions $\le t$ do not change at all. The check does exactly this, for every $t$.
:::

::: think
Q: With the mask, what is row 0 of $P$, and what is the output at position 0?
A: $[1, 0, 0, \dots]$: position 0 may only look at itself. Its output is $v_0$, its own value vector.
:::

::: think
Q: Why add the mask to the scores instead of multiplying the weights $P$ by a 0/1 mask after the softmax?
A: After multiplying, the rows would no longer add up to 1: the weight on later keys would simply disappear, and outputs at different positions would shrink by different amounts. Rescaling the rows would fix that, but adding $-10^9$ before the softmax masks and rescales in one step.
:::

::: think
Q: When the model writes text, it reads the whole context and uses only the last position's prediction. Is the mask still needed then?
A: The last position itself could see everything before it anyway. But the model was trained with the mask, so every earlier position, whose vectors the later layers use, must be computed the same way as in training. Without the mask, those vectors differ from anything the model saw during training.
:::

::: hint level=1
Build a $(T, T)$ array that is $-10^9$ strictly above the diagonal and 0 everywhere else. In `attention_single_head`, add it to the scores when `causal` is true.
:::

::: hint level=2
Start from `np.full((T, T), NEG, dtype=np.float32)` and keep the part above the diagonal with `np.triu(..., k=1)`. Add the result to the scores Tensor with `+`.
:::

::: hint level=3
```pseudo
causal_mask(T):
    return the part strictly above the diagonal of a (T, T) array full of NEG   # float32

attention_single_head(..., causal):
    ...
    if causal: scores ← scores + causal_mask(T)
    ...
```
:::

::: attempt
File: `workspace/stage2/attention.py` — fill in `causal_mask`, and use it in `attention_single_head` when `causal=True`.
Check: `python -m course check s2-l05`
:::

::: solution
@include solutions/stage2/attention.py#causal_mask

- `np.triu(..., k=1)` keeps only the entries with $j > i$. `NEG = -1e9` is finite, so $\infty - \infty$ never happens.

@include solutions/stage2/attention.py#attention_single_head

- The mask is added before the softmax; the `(T, T)` array broadcasts over the batch.
:::

::: breakit
Train the full model (s2-l09) for 300 steps with `causal=False` everywhere. The training loss falls far below anything the masked model reaches (toward 0.1), because each position can read its own target. Then let it write text: it produces garbage, because the "future" it learned to copy is not there.
:::

::: recap
- Position $i$ may only attend to $j \le i$; otherwise training leaks the answer.
- Add $0$ or $-10^9$ to the scores before the softmax: masked weights become exactly 0, and each row still adds up to 1.
- Use a large finite negative number, not $-\infty$, to avoid `nan`.
- Changing later inputs and checking earlier outputs tests the mask directly.
:::
