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
Turn the model's raw scores `logits` `(B, V)` into probabilities with softmax, and measure how wrong they are with the mean cross-entropy loss, a single number; both must survive logits as large as $10^4$.
:::

::: notation
| Symbol | Meaning |
| --- | --- |
| $z$ | a row of logits: the model's raw scores, one per vocabulary entry, shape $(V,)$ |
| $z_i$ | the score of class (character) $i$ |
| $p$ | the probabilities $\mathrm{softmax}(z)$, shape $(V,)$ |
| $p_i$ | the probability the model gives to class $i$ |
| $V$ | number of classes (vocabulary size) |
| $i, j$ | class indices, $0 \dots V-1$ |
| $c$ | any constant subtracted from every logit |
| $m$ | the largest logit in a row, $m = \max_j z_j$ |
| $y$ | the correct next character's id; $y_b$ for row $b$ |
| $L$ | the loss: mean negative log-probability of the targets |
| $B$ | batch size |
| $b$ | a row of the batch |
| $\log$ | the natural logarithm $\ln$ |
| $e$ | Euler's number, $e \approx 2.718$ |
:::

::: concept
### From scores to probabilities

The last linear layer gives each of the $V$ characters a score $z_i$, any real number. We want probabilities: positive, and summing to 1. **Softmax** makes every score positive with $e^{z_i}$ and then divides by the total:

$$
p_i = \frac{e^{z_i}}{\sum_{j=0}^{V-1} e^{z_j}}
$$

Each $p_i > 0$ because $e^{x} > 0$ for every $x$, and $\sum_i p_i = \frac{\sum_i e^{z_i}}{\sum_j e^{z_j}} = 1$. A larger score always means a larger probability, since $e^x$ is increasing.

**Worked example.** $z = [2.0, 1.0, 0.1]$:

$$
e^{2.0} = 7.389, \quad e^{1.0} = 2.718, \quad e^{0.1} = 1.105, \quad \text{sum} = 11.213
$$

$$
p = \left[\tfrac{7.389}{11.213},\ \tfrac{2.718}{11.213},\ \tfrac{1.105}{11.213}\right] = [0.659,\ 0.242,\ 0.099]
$$

### The overflow problem, and the fix

float32 overflows above about $3.4 \times 10^{38}$, which is $e^{88.7}$. A logit of 1000 gives $e^{1000} = \infty$, and $\frac{\infty}{\infty}$ is `nan`. Logits that large do happen (a badly initialized or diverging model), and one `nan` poisons every later number.

The fix is a property of softmax: subtracting the same constant $c$ from every logit changes nothing. Derivation, using $e^{a - c} = e^{a} e^{-c}$:

$$
\begin{aligned}
\frac{e^{z_i - c}}{\sum_j e^{z_j - c}}
&= \frac{e^{z_i} e^{-c}}{\sum_j e^{z_j} e^{-c}} \\
&= \frac{e^{z_i} e^{-c}}{e^{-c} \sum_j e^{z_j}} \\
&= \frac{e^{z_i}}{\sum_j e^{z_j}} = p_i
\end{aligned}
$$

Choose $c = m = \max_j z_j$. Then the largest shifted logit is exactly 0 and all others are negative, so every $e^{z_j - m}$ lies in $(0, 1]$: no overflow is possible, and the denominator is at least 1 (the term $e^{0}$), so there is no division by zero either. For $z = [1000, 1000, 999]$ the shifted logits are $[0, 0, -1]$, and softmax is $[0.422, 0.422, 0.155]$ instead of `nan`.

### Cross-entropy: how wrong is a prediction?

For one example with correct class $y$, the loss is the negative log-probability of the right answer:

$$
L = -\log p_y
$$

- $p_y = 1$ (certain and right): $L = 0$.
- $p_y = 0.5$: $L = \log 2 = 0.693$.
- $p_y \to 0$ (confident and wrong): $L \to \infty$.

In the example with $y = 0$: $L = -\log 0.659 = 0.417$. For a batch we average over the $B$ rows:

$$
L = -\frac{1}{B} \sum_{b=0}^{B-1} \log p_{b, y_b}
$$

Averaging (not summing) keeps the loss, and the learning rate that suits it, independent of the batch size.

### What loss to expect at the start

A model that knows nothing gives every character the same probability, $p_i = \frac{1}{V}$ (this is what equal logits produce: $\frac{e^{a}}{V e^{a}} = \frac{1}{V}$). Its loss is

$$
L = -\log \frac{1}{V} = \log V
$$

For $V = 65$: $\log 65 = 4.174$. A freshly initialized model should start near 4.17; much higher means it starts out confidently wrong (s1-l09 checks this).

### Log-sum-exp: a stable loss

Computing $\log(\text{softmax}(z)_y)$ is fragile: a tiny $p_y$ underflows to 0 and $\log 0 = -\infty$. Instead take the log of the definition first:

$$
\begin{aligned}
\log p_y &= \log \frac{e^{z_y}}{\sum_j e^{z_j}} \\
&= z_y - \log \sum_j e^{z_j}
\end{aligned}
$$

The second term is "log-sum-exp". Shift it by $m$ as before, using $\sum_j e^{z_j} = e^{m} \sum_j e^{z_j - m}$:

$$
\log \sum_j e^{z_j} = m + \log \sum_j e^{z_j - m}
$$

Everything inside the remaining sum is in $(0, 1]$ and the sum is at least 1, so its log is finite. Equivalently: shift the row first, $z' = z - m$, then $\log p_y = z'_y - \log \sum_j e^{z'_j}$. For logits $[10^4, 0, -10^4]$ and $y = 0$: $z' = [0, -10^4, -2 \cdot 10^4]$, the sum is $1 + e^{-10^4} + \dots = 1$ in float32, and $\log p_0 = 0$, so the loss is 0, as it should be.

**Shapes.** `logits` `(B, V)`; the row max `(B, 1)` (keep the axis so it broadcasts back over the row); log-sum-exp `(B,)`; picking $z_{b, y_b}$ for every row is integer-array indexing with two index arrays, rows `[0, 1, ..., B-1]` and columns `y`, giving `(B,)`.
:::

::: think
Q: Why does subtracting the row maximum (rather than, say, the row mean) guarantee no overflow?
A: After subtracting the max, the largest exponent is exactly $e^0 = 1$ and every other is smaller, so nothing can overflow. Subtracting the mean leaves positive values: logits $[1000, -1000]$ have mean 0, so $e^{1000}$ still overflows.
:::

::: think
Q: What should the loss be at initialization for $V = 65$, and what does a starting loss of 20 tell you?
A: About $\log 65 = 4.17$, the loss of uniform guessing. A loss of 20 means the model puts probability around $e^{-20} \approx 2 \times 10^{-9}$ on the right answers: the initial logits are large and confidently wrong, so the first steps are spent undoing a bad initialization.
:::

::: think
Q: Why average the loss over the batch instead of summing it?
A: With a sum, doubling $B$ doubles the loss and its gradient, so the right learning rate would depend on the batch size. The mean makes the loss a per-example quantity you can compare across batch sizes and between train and validation.
:::

::: hint level=1
Both functions start the same way: make each row's largest logit 0. Softmax then exponentiates and normalizes; the loss uses the "log of the definition" formula on the shifted row.
:::

::: hint level=2
`z.max(axis=-1, keepdims=True)` gives the row maxima as a column that broadcasts. `np.exp`, `.sum(axis=..., keepdims=True)`, `np.log`. To pick each row's target logit, index with two integer arrays: `np.arange(B)` for the rows and `y` for the columns. Return the loss with `float(...)`.
:::

::: hint level=3
```pseudo
softmax(logits):
    z ← logits − (row max of logits, kept as a column)
    e ← exp(z)
    return e / (row sums of e, kept as a column)

cross_entropy(logits, y):
    z     ← logits − row max                       # (B, V)
    lse   ← log(row sums of exp(z))                # (B,)
    logpy ← z[each row b, column y_b] − lse        # (B,)
    loss  ← −mean(logpy)  as a Python float
    probs ← exp(z − lse as a column)               # softmax, for backward
    return loss, (probs, y)
```
:::

::: attempt
File: `workspace/stage1/loss.py` — implement `softmax` and `cross_entropy`.
Check: `python -m course check s1-l04`
:::

::: solution
@include solutions/stage1/loss.py#softmax

- `axis=-1` makes it work for a single `(V,)` vector and for `(B, V)` batches alike; `keepdims=True` keeps the reduced axis as length 1 so the subtraction and division broadcast row by row.
- After the shift every exponent is at most 0, so `np.exp` never overflows; the sum is at least 1, so the division is safe.

@include solutions/stage1/loss.py#cross_entropy

- `lse` is the log-sum-exp of the shifted row, so `z[np.arange(B), y] - lse` is exactly $\log p_{b, y_b}$, computed without ever taking the log of a tiny probability.
- `z[np.arange(B), y]` pairs row $b$ with column $y_b$: two index arrays of length $B$ give a `(B,)` result.
- `probs = np.exp(z - lse[:, None])` reuses the work already done: $e^{z'_i - \text{lse}} = \frac{e^{z'_i}}{\sum_j e^{z'_j}}$. The backward pass in s1-l05 needs exactly these probabilities.
- `float(...)` returns a plain Python number, convenient for logging.
:::

::: breakit
Remove the max-subtraction from your softmax and feed it `np.array([[1000.0, 999.0, 0.0]], dtype=np.float32)`. Expect `RuntimeWarning: overflow encountered in exp` and an output of `[nan, nan, 0.]`. The stable version gives `[0.731, 0.269, 0.000]`.
:::

::: breakit
Compute the loss the naive way, `-np.log(softmax(z)[0, y])`, for $z = [0, -200]$ (float32) and $y = 1$. Expect `inf`: $p_1 = e^{-200}$ underflows to 0 in float32. Log-sum-exp gives the true value, 200.0.
:::

::: recap
- Softmax turns scores into probabilities; you derived why subtracting the row max changes nothing and prevents overflow.
- Cross-entropy is $-\log p_y$, averaged over the batch; uniform guessing costs $\log V$ (4.17 for $V = 65$).
- Log-sum-exp computes $\log p_y = z_y - \log \sum_j e^{z_j}$ stably, even for logits around $10^4$.
:::
