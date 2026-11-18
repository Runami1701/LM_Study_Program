---
id: s1-l04
title: Softmax and cross-entropy (stable)
stage: s1
order: 4
prereqs: [s1-l03]
files: [workspace/stage1/loss.py]
functions: [softmax, cross_entropy]
tests: tests/s1/test_l04_softmax.py
est_minutes: 40
---

::: goal
The model's last layer gives a score for every character: `logits` `(B, V)`. You will turn the scores into probabilities with softmax, and measure how wrong the probabilities are with one number, the mean cross-entropy loss. Both must work even when the scores are as large as $10^4$.
:::

::: notation
| Symbol | Meaning |
| --- | --- |
| $z$ | one row of logits: one raw score per character, shape $(V,)$ |
| $z_i$ | the score of character $i$ |
| $p$ | the probabilities $\mathrm{softmax}(z)$, shape $(V,)$ |
| $p_i$ | the probability the model gives to character $i$ |
| $V$ | number of characters (vocabulary size) |
| $i, j$ | character indices, $0 \dots V-1$ |
| $c$ | any number subtracted from every score |
| $m$ | the largest score in a row, $m = \max_j z_j$ |
| $y$ | the id of the correct next character; $y_b$ for row $b$ |
| $L$ | the loss: the mean of $-\log p_y$ over the batch |
| $B$ | batch size |
| $b$ | a row of the batch |
| $\log$ | the natural logarithm $\ln$ |
| $e$ | Euler's number, $e \approx 2.718$ |
:::

::: numpy
### One value per row, kept as a column (n-l04)

```python
z = np.array([[2.0, 1.0, 0.1],
              [0.0, 0.0, 0.0]])
print(z.max(axis=1, keepdims=True))
# [[2.]
#  [0.]]                       shape (2, 1): subtracts cleanly from z
```

### exp and log of every entry

```python
print(np.exp(np.array([0.0, 1.0, -1.0])))   # [1.         2.71828183 0.36787944]
print(np.log(np.array([1.0, np.e])))        # [0. 1.]
```

### One entry per row (n-l03)

`z[np.arange(B), y]` reads entry `y[b]` from row `b`:

```python
y = np.array([0, 2])
print(z[np.arange(2), y])                   # [2. 0.]   z[0, 0] and z[1, 2]
```

### A Python number

`float(x)` turns a NumPy number into a plain Python number.
:::

::: concept
### From scores to probabilities

The last linear layer gives each of the $V$ characters a score $z_i$, which can be any real number. We need probabilities: positive numbers that add up to 1. **Softmax** makes every score positive with $e^{z_i}$, then divides by the total:

$$
p_i = \frac{e^{z_i}}{\sum_{j=0}^{V-1} e^{z_j}}
$$

Each $p_i$ is positive, because $e^{x} > 0$ for every $x$. And $\sum_i p_i = \frac{\sum_i e^{z_i}}{\sum_j e^{z_j}} = 1$. A larger score always gives a larger probability.

**Worked example.** $z = [2.0, 1.0, 0.1]$:

$$
e^{2.0} = 7.389, \quad e^{1.0} = 2.718, \quad e^{0.1} = 1.105, \quad \text{sum} = 11.213
$$

$$
p = \left[\tfrac{7.389}{11.213},\ \tfrac{2.718}{11.213},\ \tfrac{1.105}{11.213}\right] = [0.659,\ 0.242,\ 0.099]
$$

### Huge scores overflow, and the fix

float32 cannot store numbers above about $3.4 \times 10^{38}$, which is $e^{88.7}$. A score of 1000 makes $e^{1000} = \infty$, and $\frac{\infty}{\infty}$ is `nan`. Scores that large do happen, for example in a badly started model, and one `nan` spreads to every later number.

The fix uses a property of softmax: subtracting the same number $c$ from every score changes nothing. With $e^{a - c} = e^{a} e^{-c}$:

$$
\begin{aligned}
\frac{e^{z_i - c}}{\sum_j e^{z_j - c}}
&= \frac{e^{z_i} e^{-c}}{\sum_j e^{z_j} e^{-c}} \\
&= \frac{e^{z_i} e^{-c}}{e^{-c} \sum_j e^{z_j}} \\
&= \frac{e^{z_i}}{\sum_j e^{z_j}} = p_i
\end{aligned}
$$

Choose $c = m$, the largest score in the row. Then the largest shifted score is exactly 0 and the others are negative, so every $e^{z_j - m}$ lies between 0 and 1. Nothing can overflow. The sum is at least 1 (from the $e^0$ term), so we never divide by zero either. For $z = [1000, 1000, 999]$ the shifted scores are $[0, 0, -1]$ and softmax is $[0.422, 0.422, 0.155]$ instead of `nan`.

### Cross-entropy: how wrong is a prediction?

For one example whose correct character is $y$, the loss is minus the log of the probability the model gave to $y$:

$$
L = -\log p_y
$$

- $p_y = 1$ (sure and right): $L = 0$.
- $p_y = 0.5$: $L = \log 2 = 0.693$.
- $p_y \to 0$ (sure and wrong): $L \to \infty$.

In the example with $y = 0$: $L = -\log 0.659 = 0.417$. For a batch, average over the $B$ rows:

$$
L = -\frac{1}{B} \sum_{b=0}^{B-1} \log p_{b, y_b}
$$

We take the mean rather than the sum, so the loss, and the learning rate that suits it, do not depend on the batch size.

### The loss of a model that knows nothing

A model that knows nothing gives every character the same probability, $p_i = \frac{1}{V}$. Equal scores produce exactly that: $\frac{e^{a}}{V e^{a}} = \frac{1}{V}$. Its loss is

$$
L = -\log \frac{1}{V} = \log V
$$

For $V = 65$: $\log 65 = 4.174$. A freshly started model should score about 4.17. A much higher loss means it starts out sure of wrong answers (s1-l09 checks this).

### Log-sum-exp: a safe way to compute the loss

Computing $\log(\text{softmax}(z)_y)$ directly is fragile. A tiny $p_y$ rounds to 0, and $\log 0 = -\infty$. Instead, take the log of the definition first:

$$
\begin{aligned}
\log p_y &= \log \frac{e^{z_y}}{\sum_j e^{z_j}} \\
&= z_y - \log \sum_j e^{z_j}
\end{aligned}
$$

The second term is called "log-sum-exp". Shift it by $m$ as before, using $\sum_j e^{z_j} = e^{m} \sum_j e^{z_j - m}$:

$$
\log \sum_j e^{z_j} = m + \log \sum_j e^{z_j - m}
$$

The remaining sum is at least 1, so its log is finite. In practice: shift the row first, $z' = z - m$, then $\log p_y = z'_y - \log \sum_j e^{z'_j}$. For scores $[10^4, 0, -10^4]$ and $y = 0$: $z' = [0, -10^4, -2 \cdot 10^4]$, the sum is $1$ in float32, and $\log p_0 = 0$. The loss is 0, as it should be.

**Shapes.** `logits` `(B, V)`; the row maximum `(B, 1)`; log-sum-exp `(B,)`; the correct score of each row, `z[np.arange(B), y]`, `(B,)`.
:::

::: think
Q: Why does subtracting the row maximum, and not the row mean, guarantee that nothing overflows?
A: After subtracting the maximum, the largest exponent is exactly $e^0 = 1$ and all others are smaller. Subtracting the mean can leave large positive values: scores $[1000, -1000]$ have mean 0, so $e^{1000}$ still overflows.
:::

::: think
Q: What loss should a new model with $V = 65$ have, and what does a starting loss of 20 tell you?
A: About $\log 65 = 4.17$, the loss of guessing uniformly. A loss of 20 means the model gives the right answers a probability of about $e^{-20} \approx 2 \times 10^{-9}$: its starting scores are large and confidently wrong.
:::

::: think
Q: Why average the loss over the batch instead of adding it up?
A: With a sum, doubling $B$ doubles the loss and its gradient, so the right learning rate would depend on the batch size. The mean gives a loss per example, which you can compare across batch sizes and between training and validation.
:::

::: hint level=1
Both functions start the same way: make each row's largest score 0. Softmax then exponentiates and divides by the row sum. The loss uses the "log of the definition" formula on the shifted row.
:::

::: hint level=2
`z.max(axis=-1, keepdims=True)` gives the row maxima as a column. Then `np.exp`, `.sum(axis=..., keepdims=True)` and `np.log`. To read each row's correct score, use `z[np.arange(B), y]`. Return the loss with `float(...)`.
:::

::: hint level=3
```pseudo
softmax(logits):
    z ← logits − (row maximum, kept as a column)
    e ← exp(z)
    return e / (row sums of e, kept as a column)

cross_entropy(logits, y):
    z     ← logits − row maximum                    # (B, V)
    lse   ← log(row sums of exp(z))                 # (B,)
    logpy ← z[each row b, column y_b] − lse         # (B,)
    loss  ← −mean(logpy), as a Python float
    probs ← exp(z − lse as a column)                # softmax, for the backward pass
    return loss, (probs, y)
```
:::

::: attempt
The editor below opens `workspace/stage1/loss.py`: fill in `softmax` and `cross_entropy`.
Check: press **Run check** in the Check section below.
:::

::: solution
@include solutions/stage1/loss.py#softmax

- `axis=-1` makes the function work for one `(V,)` vector and for a `(B, V)` batch. `keepdims=True` keeps the row axis, so the subtraction and division work row by row.
- After the shift every exponent is at most 0, so `np.exp` cannot overflow; the sum is at least 1, so the division is safe.

@include solutions/stage1/loss.py#cross_entropy

- `lse` is the log-sum-exp of the shifted row, so `z[np.arange(B), y] - lse` is exactly $\log p_{b, y_b}$, without ever taking the log of a tiny probability.
- `probs = np.exp(z - lse[:, None])` reuses that work: it is the softmax. The backward pass in s1-l05 needs these probabilities.
- `float(...)` returns a plain Python number.
:::

::: breakit
Remove the subtraction of the maximum from your softmax and give it `np.array([[1000.0, 999.0, 0.0]], dtype=np.float32)`. You get `RuntimeWarning: overflow encountered in exp` and the output `[nan, nan, 0.]`. The safe version gives `[0.731, 0.269, 0.000]`.
:::

::: breakit
Compute the loss the naive way, `-np.log(softmax(z)[0, y])`, for $z = [0, -200]$ (float32) and $y = 1$. You get `inf`, because $p_1 = e^{-200}$ rounds to 0 in float32. Log-sum-exp gives the true value, 200.0.
:::

::: recap
- Softmax turns scores into probabilities. Subtracting the row maximum changes nothing and prevents overflow; you derived why.
- Cross-entropy is $-\log p_y$, averaged over the batch. Guessing uniformly costs $\log V$ (4.17 for $V = 65$).
- Log-sum-exp computes $\log p_y = z_y - \log \sum_j e^{z_j}$ safely, even for scores around $10^4$.
:::
