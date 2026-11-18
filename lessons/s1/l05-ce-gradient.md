---
id: s1-l05
title: Deriving the cross-entropy gradient
stage: s1
order: 5
prereqs: [s1-l04]
files: [workspace/stage1/loss.py]
functions: [cross_entropy_backward]
tests: tests/s1/test_l05_ce_backward.py
est_minutes: 45
---

::: goal
You will derive how the loss changes when each score changes, $\frac{\partial L}{\partial z_i} = p_i - \mathbb{1}[i = y]$, and then compute it for a whole batch: `dlogits` `(B, V)`.
:::

::: notation
| Symbol | Meaning |
| --- | --- |
| $z$, $z_i$ | a row of scores and its $i$-th entry |
| $p$, $p_i$ | $\mathrm{softmax}(z)$ and its $i$-th entry |
| $y$, $y_b$ | the correct character (of row $b$) |
| $L$ | the loss: $-\log p_y$ for one example, the mean over the batch for $B$ examples |
| $S$ | the softmax denominator, $S = \sum_j e^{z_j}$ |
| $i, j$ | character indices |
| $\frac{\partial L}{\partial z_i}$ | how fast $L$ changes when only $z_i$ moves |
| $\mathbb{1}[\cdot]$ | 1 if the condition is true, otherwise 0 |
| $B$, $b$ | batch size, batch row |
| $V$ | number of characters |
| $e$ | Euler's number; $\frac{d}{dz} e^{z} = e^{z}$ |
| `dlogits` | the array of all $\frac{\partial L}{\partial z_{b,i}}$, shape `(B, V)` |
:::

::: numpy
### Copy before you change

`P.copy()` makes an independent array. Changing the copy leaves `P` alone:

```python
P = np.array([[0.7, 0.2, 0.1],
              [0.1, 0.1, 0.8]])
d = P.copy()
d[np.arange(2), np.array([0, 2])] -= 1    # subtract 1 at (0, 0) and (1, 2)
print(d)
# [[-0.3  0.2  0.1]
#  [ 0.1  0.1 -0.2]]
print(P[0])                              # [0.7 0.2 0.1]   unchanged
```

`d[rows, cols] -= 1` changes one entry per row. It is safe here because each (row, column) pair appears only once.
:::

::: concept
To train the model, we need to know, for every score, which way to move it so the loss goes down. That is the gradient $\frac{\partial L}{\partial z_i}$. We derive it for one example first, then for the batch.

### Step 1: write the loss with the scores

From s1-l04, with $S = \sum_j e^{z_j}$:

$$
L = -\log p_y = -\log \frac{e^{z_y}}{S} = -z_y + \log S
$$

(using $\log \frac{a}{b} = \log a - \log b$ and $\log e^{z_y} = z_y$). The loss is a sum of two terms, so we differentiate each one.

### Step 2: the derivative of $\log S$

By the chain rule, $\frac{\partial}{\partial z_i} \log S = \frac{1}{S} \frac{\partial S}{\partial z_i}$. In $S = e^{z_0} + e^{z_1} + \dots + e^{z_{V-1}}$, only the term $e^{z_i}$ contains $z_i$, and $\frac{d}{dz} e^{z} = e^{z}$:

$$
\frac{\partial S}{\partial z_i} = e^{z_i}
$$

$$
\frac{\partial}{\partial z_i} \log S = \frac{e^{z_i}}{S} = p_i
$$

### Step 3: the derivative of $-z_y$, two cases

- If $i = y$: $\frac{\partial}{\partial z_y}(-z_y) = -1$.
- If $i \ne y$: $-z_y$ does not contain $z_i$, so the derivative is $0$.

Both cases in one formula: $\frac{\partial}{\partial z_i}(-z_y) = -\mathbb{1}[i = y]$.

### Step 4: add the two parts

$$
\frac{\partial L}{\partial z_i} = p_i - \mathbb{1}[i = y]
$$

Case by case:

$$
\frac{\partial L}{\partial z_y} = p_y - 1 \le 0, \qquad \frac{\partial L}{\partial z_i} = p_i \ge 0 \quad (i \ne y)
$$

**What this means.** Training moves each score *against* its gradient. So the correct score goes **up**, by an amount that grows with $1 - p_y$, the distance from certainty. Every wrong score goes **down**, in proportion to the probability it wrongly received. A perfect prediction ($p_y = 1$, every other $p_i = 0$) has gradient zero: there is nothing left to fix.

**Worked example.** $z = [2.0, 1.0, 0.1]$ and $y = 0$, so $p = [0.659, 0.242, 0.099]$ (from s1-l04):

$$
\frac{\partial L}{\partial z} = [0.659 - 1,\ 0.242,\ 0.099] = [-0.341,\ 0.242,\ 0.099]
$$

Check it: raise $z_0$ from 2.0 to 2.001. The loss should drop by about $0.341 \times 0.001 = 0.000341$, from 0.41703 to about 0.41669. Recomputing the loss gives exactly that.

### Step 5: the batch mean

The batch loss is $L = \frac{1}{B} \sum_b L_b$, and the scores of row $b$ appear only in $L_b$. So

$$
\frac{\partial L}{\partial z_{b,i}} = \frac{1}{B} \frac{\partial L_b}{\partial z_{b,i}} = \frac{1}{B}\left(p_{b,i} - \mathbb{1}[i = y_b]\right)
$$

In code: take `probs` `(B, V)`, subtract 1 at position $(b, y_b)$ in every row, and divide everything by $B$.

### A built-in check

Every row of `dlogits` adds up to zero:

$$
\sum_i \left(p_i - \mathbb{1}[i = y]\right) = \sum_i p_i - 1 = 1 - 1 = 0
$$

This follows from s1-l04: adding the same number to every score changes no probability, so it cannot change the loss.
:::

::: think
Q: Why does each row of `dlogits` add up to zero? Answer without algebra.
A: Raising all scores of a row by the same amount changes no probability, so the loss stays the same. The sum of the row's gradient is exactly the rate of change for that move, so it is 0.
:::

::: think
Q: The model gives the correct character $p_y = 0.99$. How large are the gradients in that row, and what does that mean for learning?
A: $\frac{\partial L}{\partial z_y} = -0.01$, and the wrong scores share $+0.01$ in total. These are tiny (and divided by $B$ as well). Examples the model already gets right contribute almost nothing; training concentrates on the mistakes.
:::

::: think
Q: If you forget the $\frac{1}{B}$, what happens when you double the batch size?
A: The gradient roughly doubles, because it becomes a sum over twice as many rows. The steps double too, so a learning rate that worked at $B = 32$ may blow up at $B = 64$.
:::

::: hint level=1
`cross_entropy` already saved the probabilities and `y` in its cache. The gradient is those probabilities with one entry per row changed, then divided by the batch size.
:::

::: hint level=2
Copy the probabilities with `.copy()`, so the cache stays unchanged. Subtract 1 at each row's correct column with `d[np.arange(B), y] -= 1`. Then divide by `B`.
:::

::: hint level=3
```pseudo
cross_entropy_backward(cache):
    probs, y ← cache
    B        ← number of rows of probs
    d        ← a copy of probs                    # (B, V)
    d[row b, column y_b] −= 1   for every b
    return d / B
```
:::

::: attempt
File: `workspace/stage1/loss.py` — fill in `cross_entropy_backward`.
Check: `python -m course check s1-l05`
:::

::: solution
@include solutions/stage1/loss.py#cross_entropy_backward

- `probs.copy()`: the cache belongs to the forward pass. Changing it in place would break a second call of the backward pass.
- `dlogits[np.arange(B), y] -= 1.0` subtracts $\mathbb{1}[i = y_b]$: one entry per row.
- `/ B` is the factor from the batch mean. Dividing a float32 array by a Python int keeps float32.
:::

::: breakit
Remove the `/ B`. Then compute the gradient for one row of scores repeated 1, 8 and 64 times (`np.repeat(z, n, axis=0)`) and add up the rows. The sum grows 8 and 64 times; with `/ B` it stays the same. If you train with this bug in s1-l10 at $B = 128$, the loss becomes `nan` within a few hundred steps, at a learning rate that works fine with the correct gradient.
:::

::: recap
- You derived $\frac{\partial L}{\partial z_i} = p_i - \mathbb{1}[i = y]$ from $L = -z_y + \log \sum_j e^{z_j}$, case by case.
- The batch mean adds a factor $\frac{1}{B}$.
- Each row of `dlogits` adds up to 0: the correct score is pulled up by $1 - p_y$, each wrong score pushed down by its $p_i$.
:::
