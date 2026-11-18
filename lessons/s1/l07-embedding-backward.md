---
id: s1-l07
title: Embedding backward
stage: s1
order: 7
prereqs: [s1-l06]
files: [workspace/stage1/layers.py]
functions: [embedding_backward]
tests: tests/s1/test_l07_embedding_backward.py
est_minutes: 35
---

::: goal
Send the gradient `demb` `(B, k, e)` back into the embedding table: `dC` `(V, e)`, where every looked-up row receives the sum of the gradients from all the places it was used.
:::

::: notation
| Symbol | Meaning |
| --- | --- |
| $C$ | embedding table, shape $(V, e)$; row $C_r$ is the vector of id $r$ |
| $C_{rc}$ | entry of $C$: row (id) $r$, column $c$ |
| $x$ | the ids looked up, shape $(B, k)$; $x_{bj}$ is the id at batch row $b$, context position $j$ |
| $\text{emb}$ | the lookup result, $\text{emb}_{bjc} = C_{x_{bj}, c}$, shape $(B, k, e)$ |
| $\text{demb}$ | $\frac{\partial L}{\partial \text{emb}}$, shape $(B, k, e)$ |
| $dC$ | $\frac{\partial L}{\partial C}$, shape $(V, e)$ |
| $B, k, e, V$ | batch size, context length, embedding size, vocabulary size |
| $b, j, c, r$ | indices: batch row, context position, embedding column, table row |
| $L$ | the loss |
| $\mathbb{1}[\cdot]$ | 1 if the condition holds, else 0 |
:::

::: concept
### Which rows get a gradient?

The forward pass copied rows: $\text{emb}_{bjc} = C_{x_{bj}, c}$. So entry $C_{rc}$ influences $\text{emb}_{bjc}$ exactly when $x_{bj} = r$, with local derivative 1. Summing over all paths (s1-l06):

$$
\begin{aligned}
\frac{\partial L}{\partial C_{rc}}
&= \sum_{b} \sum_{j} \frac{\partial L}{\partial \text{emb}_{bjc}} \cdot \frac{\partial \text{emb}_{bjc}}{\partial C_{rc}} \\
&= \sum_{b} \sum_{j} \text{demb}_{bjc} \cdot \mathbb{1}[x_{bj} = r]
\end{aligned}
$$

In words: **row $r$ of $dC$ is the sum of the `demb` vectors at every position where id $r$ was used.** Rows of ids that do not appear in the batch get exactly zero.

This is the same lesson as `db` in s1-l06: one parameter used in several places collects one gradient per use, and they add.

**Worked example.** $V = 3$, $e = 2$, one example with context `x = [[1, 1]]` (the same character twice), and

$$
\text{demb} = \begin{bmatrix} 1 & 2 \\ 10 & 20 \end{bmatrix} \quad \text{(positions } j = 0, 1\text{)}
$$

Row 1 was used at both positions, so $dC_1 = [1, 2] + [10, 20] = [11, 22]$, and

$$
dC = \begin{bmatrix} 0 & 0 \\ 11 & 22 \\ 0 & 0 \end{bmatrix}
$$

### The duplicate-index trap

The natural guess is "the backward of `C[x]` is `dC[x] += demb`". Try it on the example. NumPy evaluates `dC[x] += demb` as **one** read, one add, one write:

1. read `dC[[1, 1]]`: two copies of row 1, both zero;
2. add `demb`: gives `[1, 2]` and `[10, 20]`;
3. write both results to row 1: the second write overwrites the first.

Row 1 ends up as `[10, 20]`: the contribution `[1, 2]` silently vanished. No error, no warning, just a wrong gradient. The fancy-index assignment is **buffered**: each target is written once per statement, no matter how often it appears in the index.

In real batches repeated ids are everywhere (`e` appears in nearly every 8-character context), so this bug would corrupt the gradient of the most common characters most.

### The fix: an unbuffered add

`np.add.at(dC, x, demb)` performs the additions one at a time ("unbuffered"), so repeated indices accumulate. Shapes: `dC` `(V, e)`, index `x` `(B, k)`, values `demb` `(B, k, e)`: for each position $(b, j)$ it adds the length-$e$ vector `demb[b, j]` to row `x[b, j]`.

### Un-flattening first

In the model, the gradient arrives at the flatten step as `dflat` `(B, k·e)`. Flattening was a reshape, which only relabels positions, so its backward is the reverse reshape: `(B, k·e)` → `(B, k, e)`. Then `embedding_backward` takes over. (You will wire this up in s1-l09.)
:::

::: think
Q: In the worked example, what would `dC[x] += demb` produce, and why is no error raised?
A: Row 1 becomes `[10, 20]` instead of `[11, 22]`. The statement is legal NumPy: it reads `dC[x]`, adds, and assigns back; duplicate targets in an assignment just mean the last write wins. Nothing is "wrong" from NumPy's point of view.
:::

::: think
Q: Why does the backward of a reshape need no math, only another reshape?
A: A reshape moves no numbers and changes none: element $(b, j, c)$ and element $(b, j \cdot e + c)$ are the same number. So each gradient entry just goes back to the label it came from, which is the inverse reshape.
:::

::: think
Q: In a batch of $B = 128$ contexts of $k = 8$ characters, about how many times is the space character (≈15% of the text) used?
A: About $0.15 \times 128 \times 8 \approx 154$ times. Its row of $dC$ is the sum of ~154 vectors; with `dC[x] += demb` it would get only one of them.
:::

::: hint level=1
Start from an all-zero table of the right shape and add each position's gradient vector into the row of the id used there. Be careful how you add when ids repeat.
:::

::: hint level=2
`np.zeros((V, e), dtype=demb.dtype)` for the start; `np.add.at(target, indices, values)` is the unbuffered add that accumulates repeated indices. It accepts a `(B, k)` index array and `(B, k, e)` values directly.
:::

::: hint level=3
```pseudo
embedding_backward(demb, x, V):
    e  ← last dimension of demb
    dC ← zeros (V, e), same dtype as demb
    for every position (b, j), unbuffered:  dC[x[b, j]] += demb[b, j]
    return dC
```
:::

::: attempt
File: `workspace/stage1/layers.py` — implement `embedding_backward`.
Check: `python -m course check s1-l07`
:::

::: solution
@include solutions/stage1/layers.py#embedding_backward

- `dtype=demb.dtype` keeps float32 in training and float64 during gradient checks.
- `np.add.at(dC, x, demb)` is the vectorized form of "for each position, add its vector into the row it used", and it accumulates duplicates.
- Rows of ids that never appear stay zero, as the derivation says.
:::

::: breakit
Replace `np.add.at(dC, x, demb)` with `dC[x] += demb` and run the check. Expect `test_repeated_ids_accumulate` to fail with row 1 equal to `[10, 20]` instead of `[11, 22]`, while a check on a batch without repeated ids would still pass. That is why the test uses a duplicate on purpose.
:::

::: recap
- Each table row's gradient is the sum of the gradients at every position that used it; unused rows get zero.
- `dC[x] += demb` is buffered and silently drops duplicates; `np.add.at` accumulates.
- The backward of flatten is the inverse reshape.
:::
