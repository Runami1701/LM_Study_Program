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
Derive, line by line, how the loss changes when each logit changes, $\frac{\partial L}{\partial z_i} = p_i - \mathbb{1}[i = y]$, and implement it for a batch: `dlogits` `(B, V)`.
:::

::: notation
| Symbol | Meaning |
| --- | --- |
| $z$, $z_i$ | a row of logits and its $i$-th entry |
| $p$, $p_i$ | $\mathrm{softmax}(z)$ and its $i$-th entry |
| $y$, $y_b$ | the target class (of row $b$) |
| $L$ | the loss: $-\log p_y$ for one example, the mean over the batch for $B$ examples |
| $S$ | the softmax denominator $S = \sum_j e^{z_j}$ |
| $i, j$ | class indices |
| $\frac{\partial L}{\partial z_i}$ | partial derivative: how fast $L$ changes when only $z_i$ moves |
| $\mathbb{1}[\cdot]$ | indicator: 1 if the condition is true, else 0 |
| $B$, $b$ | batch size, batch row |
| $V$ | number of classes |
| $e$ | Euler's number; $\frac{d}{dz} e^{z} = e^{z}$ |
| `dlogits` | the array of all $\frac{\partial L}{\partial z_{b,i}}$, shape `(B, V)` |
:::

::: concept
Training needs to know, for every logit, which way to nudge it to lower the loss. That is the gradient $\frac{\partial L}{\partial z_i}$. We derive it for one example, then for the batch mean.

### Step 1: the loss in terms of the logits

From s1-l04, with $S = \sum_j e^{z_j}$:

$$
L = -\log p_y = -\log \frac{e^{z_y}}{S} = -z_y + \log S
$$

(using $\log \frac{a}{b} = \log a - \log b$ and $\log e^{z_y} = z_y$). Two terms, so we differentiate each.

### Step 2: the derivative of $\log S$

The chain rule says $\frac{\partial}{\partial z_i} \log S = \frac{1}{S} \frac{\partial S}{\partial z_i}$. In $S = e^{z_0} + e^{z_1} + \dots + e^{z_{V-1}}$, only the term $e^{z_i}$ depends on $z_i$, and $\frac{d}{dz} e^{z} = e^{z}$:

$$
\frac{\partial S}{\partial z_i} = e^{z_i}
$$

$$
\frac{\partial}{\partial z_i} \log S = \frac{e^{z_i}}{S} = p_i
$$

### Step 3: the derivative of $-z_y$, two cases

- If $i = y$: $\frac{\partial}{\partial z_y}(-z_y) = -1$.
- If $i \ne y$: $-z_y$ does not contain $z_i$, so the derivative is $0$.

Both cases at once: $\frac{\partial}{\partial z_i}(-z_y) = -\mathbb{1}[i = y]$.

### Step 4: combine

$$
\frac{\partial L}{\partial z_i} = p_i - \mathbb{1}[i = y]
$$

Case by case:

$$
\frac{\partial L}{\partial z_y} = p_y - 1 \le 0, \qquad \frac{\partial L}{\partial z_i} = p_i \ge 0 \quad (i \ne y)
$$

**Intuition.** Gradient descent moves each logit *against* its gradient. So the correct logit goes **up** by an amount proportional to $1 - p_y$ (how far it is from certain), and every wrong logit goes **down** in proportion to the probability it wrongly received. A perfect prediction ($p_y = 1$, all other $p_i = 0$) gives a zero gradient: nothing left to fix.

**Worked example.** $z = [2.0, 1.0, 0.1]$, $y = 0$, so $p = [0.659, 0.242, 0.099]$ (s1-l04):

$$
\frac{\partial L}{\partial z} = [0.659 - 1,\ 0.242,\ 0.099] = [-0.341,\ 0.242,\ 0.099]
$$

Check it numerically: raise $z_0$ from 2.0 to 2.001. The loss should fall by about $0.341 \times 0.001 = 0.000341$, from 0.41703 to about 0.41669, which is what you get by recomputing.

### Step 5: the batch mean

The batch loss is $L = \frac{1}{B} \sum_b L_b$, and row $b$'s logits appear only in $L_b$. So

$$
\frac{\partial L}{\partial z_{b,i}} = \frac{1}{B} \frac{\partial L_b}{\partial z_{b,i}} = \frac{1}{B}\left(p_{b,i} - \mathbb{1}[i = y_b]\right)
$$

As an array: take `probs` `(B, V)`, subtract 1 at position $(b, y_b)$ in every row, divide everything by $B$.

### A built-in sanity check

Each row of `dlogits` sums to zero:

$$
\sum_i \left(p_i - \mathbb{1}[i = y]\right) = \sum_i p_i - 1 = 1 - 1 = 0
$$

This is softmax's shift invariance seen from the gradient: adding the same amount to every logit leaves the probabilities, and so the loss, unchanged, so the gradient has no component along "all logits together".
:::

::: think
Q: Why does each row of `dlogits` sum to zero? Answer without algebra.
A: Raising all logits of a row by the same amount changes no probability (shift invariance), so the loss doesn't change. The sum of the row's gradient entries is exactly the rate of change for that "raise everything" move, so it must be 0.
:::

::: think
Q: The model gives the right class $p_y = 0.99$. How big are the gradients on that row, and what does that mean for learning?
A: $\frac{\partial L}{\partial z_y} = -0.01$ and the wrong logits share $+0.01$ in total: tiny. Examples the model already gets right contribute almost nothing; training focuses on what it gets wrong. (Divided by $B$ on top.)
:::

::: think
Q: If you forget the $\frac{1}{B}$, what happens when you double the batch size?
A: The gradient roughly doubles (it becomes a sum over twice as many rows), so the effective step doubles too. A learning rate tuned at $B = 32$ might diverge at $B = 64$.
:::

::: hint level=1
You already computed the probabilities in `cross_entropy` and saved them in the cache, along with `y`. The gradient is those probabilities with one entry per row changed, then scaled.
:::

::: hint level=2
Copy the probabilities (don't modify the cache in place). Use integer-array indexing with `np.arange(B)` and `y` to subtract 1 at each row's target position, then divide by `B`.
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
File: `workspace/stage1/loss.py` — implement `cross_entropy_backward`.
Check: `python -m course check s1-l05`
:::

::: solution
@include solutions/stage1/loss.py#cross_entropy_backward

- `probs.copy()`: the cache belongs to the forward pass; modifying it in place would corrupt it if backward were ever called twice.
- `dlogits[np.arange(B), y] -= 1.0` subtracts $\mathbb{1}[i = y_b]$: exactly one entry per row.
- `/ B` is the batch mean's factor. Dividing a float32 array by a Python int keeps float32.
:::

::: breakit
Drop the `/ B`, then compute the gradient for the same row repeated 1, 8 and 64 times (`np.repeat(z, n, axis=0)`). Expect the summed gradient to grow 8× and 64×; with `/ B` it stays the same. In s1-l10, train with this bug and $B = 128$: expect the loss to blow up to `nan` within a few hundred steps at a learning rate that is fine with the correct gradient.
:::

::: recap
- You derived $\frac{\partial L}{\partial z_i} = p_i - \mathbb{1}[i = y]$ from $L = -z_y + \log \sum_j e^{z_j}$, case by case.
- The batch mean adds a factor $\frac{1}{B}$.
- Rows of `dlogits` sum to zero; correct classes are pulled up by $1 - p_y$, wrong ones pushed down by $p_i$.
:::
