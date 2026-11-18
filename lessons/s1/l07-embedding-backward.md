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
You will send the gradient `demb` `(B, k, e)` back into the embedding table, giving `dC` `(V, e)`. Every row that was looked up receives the sum of the gradients from all the places where it was used.
:::

::: notation
| Symbol | Meaning |
| --- | --- |
| $C$ | embedding table, shape $(V, e)$; row $C_r$ is the vector of id $r$ |
| $C_{rc}$ | entry of $C$: row (id) $r$, column $c$ |
| $x$ | the ids that were looked up, shape $(B, k)$; $x_{bj}$ is the id at batch row $b$, position $j$ |
| $\text{emb}$ | the lookup result, $\text{emb}_{bjc} = C_{x_{bj}, c}$, shape $(B, k, e)$ |
| $\text{demb}$ | $\frac{\partial L}{\partial \text{emb}}$, shape $(B, k, e)$ |
| $dC$ | $\frac{\partial L}{\partial C}$, shape $(V, e)$ |
| $B, k, e, V$ | batch size, context length, embedding size, vocabulary size |
| $b, j, c, r$ | indices: batch row, context position, embedding column, table row |
| $L$ | the loss |
| $\mathbb{1}[\cdot]$ | 1 if the condition holds, otherwise 0 |
:::

::: numpy
### Adding whole vectors into chosen rows (n-l03)

`np.add.at(target, ids, values)` adds `values[...]` into `target[ids[...]]`, one position at a time, so repeated ids add up. It also works when `ids` is a 2-D array and each value is a vector:

```python
dC = np.zeros((3, 2))                    # V = 3 rows, e = 2
ids = np.array([[1, 1]])                 # (B=1, k=2): id 1 used twice
g = np.array([[[1., 2.], [10., 20.]]])   # (B=1, k=2, e=2): one vector per position
np.add.at(dC, ids, g)
print(dC)
# [[ 0.  0.]
#  [11. 22.]                             both vectors added into row 1
#  [ 0.  0.]]
```
:::

::: concept
### Which rows get a gradient?

The forward pass copied rows: $\text{emb}_{bjc} = C_{x_{bj}, c}$. So entry $C_{rc}$ affects $\text{emb}_{bjc}$ exactly when $x_{bj} = r$, with local derivative 1. Adding up all paths (s1-l06):

$$
\begin{aligned}
\frac{\partial L}{\partial C_{rc}}
&= \sum_{b} \sum_{j} \frac{\partial L}{\partial \text{emb}_{bjc}} \cdot \frac{\partial \text{emb}_{bjc}}{\partial C_{rc}} \\
&= \sum_{b} \sum_{j} \text{demb}_{bjc} \cdot \mathbb{1}[x_{bj} = r]
\end{aligned}
$$

In words: **row $r$ of $dC$ is the sum of the `demb` vectors at every position where id $r$ was used.** Ids that do not appear in the batch get a row of zeros.

This is the same idea as `db` in s1-l06: a parameter used in several places collects one gradient per use, and they add up.

**Worked example.** $V = 3$, $e = 2$, one example with context `x = [[1, 1]]` (the same character twice), and

$$
\text{demb} = \begin{bmatrix} 1 & 2 \\ 10 & 20 \end{bmatrix} \quad \text{(positions } j = 0, 1\text{)}
$$

Row 1 was used at both positions, so $dC_1 = [1, 2] + [10, 20] = [11, 22]$, and

$$
dC = \begin{bmatrix} 0 & 0 \\ 11 & 22 \\ 0 & 0 \end{bmatrix}
$$

### The trap: repeated ids

The obvious guess is "the backward of `C[x]` is `dC[x] += demb`". Try it on the example. NumPy runs `dC[x] += demb` as **one** read, one addition and one write:

1. read `dC[[1, 1]]`: two copies of row 1, both zero;
2. add `demb`: this gives `[1, 2]` and `[10, 20]`;
3. write both results into row 1: the second write overwrites the first.

Row 1 ends up as `[10, 20]`, and the contribution `[1, 2]` is gone. NumPy raises no error and no warning; the gradient is simply wrong.

Real batches repeat ids constantly: the space character is about 15% of the text, so it appears about $0.15 \times 128 \times 8 \approx 154$ times in a batch of 128 contexts of 8 characters. This bug would hurt exactly the most common characters.

### The fix: `np.add.at`

`np.add.at(dC, x, demb)` performs the additions one at a time, so repeated ids add up (see the box above). Shapes: `dC` `(V, e)`, `x` `(B, k)`, `demb` `(B, k, e)`. For each position $(b, j)$ it adds the vector `demb[b, j]` into row `x[b, j]`.

### Un-flattening first

In the model, the gradient reaches the flatten step as `dflat` `(B, k·e)`. Flattening was a `reshape`, which only relabels the numbers, so its backward is the reverse reshape: `(B, k·e)` → `(B, k, e)`. Then `embedding_backward` takes over. You connect these steps in s1-l09.
:::

::: think
Q: In the worked example, what does `dC[x] += demb` produce, and why does NumPy raise no error?
A: Row 1 becomes `[10, 20]` instead of `[11, 22]`. The statement is valid NumPy: it reads `dC[x]`, adds, and writes back. When two writes go to the same row, the last one wins.
:::

::: think
Q: Why does the backward of a reshape need no math, only another reshape?
A: A reshape neither moves nor changes any number: entry $(b, j, c)$ and entry $(b, j \cdot e + c)$ are the same number. So each gradient entry goes back to the label it came from, which is the reverse reshape.
:::

::: hint level=1
Start from a table of zeros with the right shape. Add each position's gradient vector into the row of the id used there. Be careful with repeated ids.
:::

::: hint level=2
`np.zeros((V, e), dtype=demb.dtype)` for the start; `np.add.at(dC, x, demb)` adds with repeats counted. It accepts the `(B, k)` id array and the `(B, k, e)` values directly.
:::

::: hint level=3
```pseudo
embedding_backward(demb, x, V):
    e  ← last size of demb's shape
    dC ← zeros (V, e), same dtype as demb
    for every position (b, j), one at a time:  dC[x[b, j]] += demb[b, j]
    return dC
```
:::

::: attempt
The editor below opens `workspace/stage1/layers.py`: fill in `embedding_backward`.
Check: press **Run check** in the Check section below.
:::

::: solution
@include solutions/stage1/layers.py#embedding_backward

- `dtype=demb.dtype` keeps float32 in training and float64 in gradient checks.
- `np.add.at(dC, x, demb)` is "for each position, add its vector into the row it used", done without a Python loop, and with repeats counted.
- Ids that never appear keep a row of zeros, as the derivation says.
:::

::: breakit
Replace `np.add.at(dC, x, demb)` with `dC[x] += demb` and run the check. `test_repeated_ids_accumulate` fails: row 1 is `[10, 20]` instead of `[11, 22]`. A check on a batch without repeated ids would still pass, which is why the test uses a repeated id on purpose.
:::

::: recap
- Each table row's gradient is the sum of the gradients at every position that used it; unused rows get zeros.
- `dC[x] += demb` silently drops repeats; `np.add.at` adds them all.
- The backward of flatten is the reverse reshape.
:::
