---
id: s2-l04
title: Single-head attention
stage: s2
order: 4
prereqs: [s2-l03]
files: [workspace/stage2/attention.py]
functions: [softmax, attention_single_head]
tests: tests/s2/test_l04_attention.py
est_minutes: 75
---

::: goal
You will let every position collect information from the other positions. Each position gets a query, a key and a value `(B, T, d_h)`; the scores $QK^\top / \sqrt{d_h}$ `(B, T, T)` go through a softmax over the keys; and the output is the weighted sum $PV$ `(B, T, d_h)`. You derive why the scores are divided by $\sqrt{d_h}$, and the backward pass of softmax.
:::

::: notation
| Symbol | Meaning |
| --- | --- |
| $x$ | input, $(B, T, d)$ |
| $W_q, W_k, W_v$ | projection matrices, $(d, d_h)$ each |
| $Q, K, V$ | queries $xW_q$, keys $xW_k$, values $xW_v$: $(B, T, d_h)$ each |
| $d_h$ | head size: the length of each query, key and value vector |
| $S$ | scores $QK^\top/\sqrt{d_h}$, $(B, T, T)$; $S_{ij}$ compares query $i$ with key $j$ |
| $P$ | attention weights $\mathrm{softmax}(S)$ over the last axis, $(B, T, T)$ |
| $q, k$ | one query and one key vector, length $d_h$ |
| $q_m, k_m$ | their $m$-th entries |
| $i, j, m$ | query position, key position, vector entry |
| $\mathrm{Var}$, $\mathbb{E}$ | variance, expected value |
| $dP, dS$ | gradients of the loss with respect to $P$ and $S$ |
| $\delta_{ij}$ | 1 if $i = j$, otherwise 0 |
| $\odot$ | entry-by-entry product |
| $\top$ | transpose of the last two axes |
| $p, s$ | one row of $P$ and of $S$ |
| $L$ | the loss |
| $e$ | Euler's number |
:::

::: numpy
### Batched matrix products

`@` on 3-D arrays multiplies matching matrices of the batch: `(B, n, m) @ (B, m, p)` gives `(B, n, p)`. To compute $QK^\top$, swap the last two axes of $K$ first (n-l02):

```python
q = np.zeros((32, 128, 32))           # (B, T, d_h)
k = np.zeros((32, 128, 32))
print((q @ np.transpose(k, (0, 2, 1))).shape)   # (32, 128, 128)   one T x T table per sequence
```

On your engine the same step is `q @ ops.transpose(k, (0, 2, 1))`.

### A row sum kept as a column (n-l04)

The softmax backward needs `(g * P).sum(axis=-1, keepdims=True)`: one number per row, shaped so it subtracts from every entry of that row.
:::

::: concept
**See it drawn:** the <a href="explainers/matrices-attention.html#attention" target="_blank" rel="noopener">visual companion</a>, sections 4 to 6: dot products, attention step by step with the causal mask, and the √dₕ scale. Click the cells and tokens; every number is recomputed live.

### The idea in one sentence

Each position asks a question (its **query** $q_i$), every position advertises what it holds (its **key** $k_j$), and position $i$ collects the **values** $v_j$ of the positions whose keys match its query best.

### Step by step, with shapes

| step | formula | shape |
| --- | --- | --- |
| project | $Q = xW_q$, $K = xW_k$, $V = xW_v$ | $(B, T, d_h)$ each |
| compare | $S = QK^\top / \sqrt{d_h}$ | $(B, T, d_h)(B, d_h, T) = (B, T, T)$ |
| normalize | $P = \mathrm{softmax}(S)$ along the last (key) axis | $(B, T, T)$; each row adds up to 1 |
| mix | $\text{out} = PV$ | $(B, T, T)(B, T, d_h) = (B, T, d_h)$ |

Row $i$ of $P$ is a probability distribution over the positions. Output $i$ is the average of the value vectors, weighted by that row.

### Why divide by $\sqrt{d_h}$

Take one query and one key whose entries are independent, each with mean 0 and variance 1. (Inputs after LayerNorm and small random projections behave roughly like this.) Their dot product is a sum of $d_h$ terms:

$$
q \cdot k = \sum_{m=1}^{d_h} q_m k_m
$$

Each term has mean $\mathbb{E}[q_m]\mathbb{E}[k_m] = 0$ and variance $\mathbb{E}[q_m^2]\mathbb{E}[k_m^2] = 1 \cdot 1 = 1$. Variances of independent terms add:

$$
\mathrm{Var}(q \cdot k) = \sum_{m=1}^{d_h} \mathrm{Var}(q_m k_m) = d_h
$$

So the raw scores have standard deviation $\sqrt{d_h}$: about 5.7 for $d_h = 32$, and 16 for $d_h = 256$. A softmax of numbers that far apart is almost one-hot: one key gets nearly all the weight, and the others get almost 0 weight and almost 0 gradient. The softmax is saturated, like tanh in s1-l03. Dividing by $\sqrt{d_h}$ brings the variance back to $\frac{d_h}{d_h} = 1$, whatever the head size.

### Softmax backward, derived

For one row, $p_i = \frac{e^{s_i}}{\sum_k e^{s_k}}$. By the quotient rule:

$$
\begin{aligned}
\frac{\partial p_i}{\partial s_j}
&= \frac{\delta_{ij} e^{s_i} \sum_k e^{s_k} - e^{s_i} e^{s_j}}{\left(\sum_k e^{s_k}\right)^2} \\
&= p_i\,\delta_{ij} - p_i\,p_j = p_i(\delta_{ij} - p_j)
\end{aligned}
$$

Every $p_i$ depends on every $s_j$, so sum over $i$:

$$
\begin{aligned}
dS_j &= \sum_i dP_i\, p_i (\delta_{ij} - p_j) \\
&= dP_j\, p_j - p_j \sum_i dP_i\, p_i \\
&= p_j \left(dP_j - \sum_i dP_i\, p_i\right)
\end{aligned}
$$

For every row at once:

$$
dS = P \odot \left(dP - \mathrm{rowsum}(dP \odot P)\right)
$$

(the row sum is kept as a column, so it subtracts from every entry of its row). This is cheaper and safer than building softmax from `exp`, `sum` and division operations, so softmax becomes one operation, like cross-entropy.

**Three-token example.** One row of scores $s = [1, 0, -1]$ gives $p = [0.665, 0.245, 0.090]$. If the gradient arriving from above is $dP = [1, 0, 0]$, then $\sum_i dP_i\,p_i = 0.665$ and

$$
dS = [0.665 \cdot 0.335,\ 0.245 \cdot (-0.665),\ 0.090 \cdot (-0.665)] = [0.223,\ -0.163,\ -0.060]
$$

The entries add up to 0, because adding the same number to every score changes no weight (s1-l04).
:::

::: think
Q: With $d_h = 64$ and entries of variance 1, what is the standard deviation of a raw score, and what happens to a softmax over 128 keys?
A: $\sqrt{64} = 8$. The scores spread over roughly $\pm 16$. After the exponential, the largest wins by factors like $e^{8}$, so each row is almost one-hot. The gradients through the other keys vanish, and the model learns slowly.
:::

::: think
Q: Why does each row of $dS$ always add up to zero?
A: Adding the same number to a row of scores leaves the softmax unchanged, so the loss does not change. The row sum of $dS$ is the rate of change in that direction, so it is 0. In symbols: $\sum_j p_j(dP_j - c) = c - c = 0$ with $c = \sum_i dP_i p_i$.
:::

::: think
Q: Write the shapes of both matrix products for $B = 32$, $T = 128$, $d_h = 32$. Which array is the largest?
A: $QK^\top$: `(32, 128, 32) @ (32, 32, 128) → (32, 128, 128)`. $PV$: `(32, 128, 128) @ (32, 128, 32) → (32, 128, 32)`. The scores and weights, `(32, 128, 128)`, are the largest: 524K numbers each, and they grow with $T^2$.
:::

::: hint level=1
`softmax` is the safe softmax of s1-l04 plus a backward function that applies the boxed formula. `attention_single_head` is four lines of engine operations: three projections, the scores, the softmax, and the mix.
:::

::: hint level=2
`ops.transpose(k, (0, 2, 1))` gives $K^\top$. Multiply the scores by `1.0 / math.sqrt(d_h)`. (`1.0 / np.sqrt(d_h)` is also fine here: the engine's `*` converts the number to the data's dtype.) In the softmax backward: `(g * P).sum(axis=-1, keepdims=True)`.
:::

::: hint level=3
```pseudo
softmax(x):
    P ← safe softmax of x.data over the last axis
    out ← Tensor(P, parents (x,))
    backward: x.grad += P · (g − rowsum(g · P))

attention_single_head(x, Wq, Wk, Wv, causal):
    q, k, v ← x @ Wq, x @ Wk, x @ Wv                         # (B, T, d_h)
    scores ← (q @ transpose(k, (0, 2, 1))) · (1 / sqrt(d_h))   # (B, T, T)
    (s2-l05: if causal, add the mask)
    return softmax(scores) @ v                                # (B, T, d_h)
```
:::

::: attempt
File: `workspace/stage2/attention.py` — fill in `softmax` and `attention_single_head`. Ignore `causal` until s2-l05.
Check: `python -m course check s2-l04`
:::

::: solution
@include solutions/stage2/attention.py#softmax

- Forward: subtract the maximum, as in s1-l04, for any number of front axes.
- Backward: the derived $P \odot (dP - \mathrm{rowsum}(dP \odot P))$, using the saved `P`.

@include solutions/stage2/attention.py#attention_single_head

- The three projections are engine matrix products, so their weights get gradients automatically.
- Multiplying by a Python number goes through `Tensor.__mul__`, which converts it to the data's dtype, so float32 is kept.
- The `causal` branch belongs to s2-l05.
:::

::: breakit
Remove the $1/\sqrt{d_h}$. Run attention with identity matrices as projections on random inputs of variance 1 with $d_h = 256$, and print the largest entry of each row of $P$. Most rows have a maximum above 0.99 (one-hot attention). With the scale, typical maxima are about 0.2 to 0.5.
:::

::: recap
- Attention: queries match keys, softmax turns the matches into weights, and the outputs average the values.
- Scores grow like $\sqrt{d_h}$ (you derived $\mathrm{Var}(q \cdot k) = d_h$); dividing by it keeps the softmax soft.
- Softmax backward: $\frac{\partial p_i}{\partial s_j} = p_i(\delta_{ij} - p_j)$, so $dS = P \odot (dP - \mathrm{rowsum}(dP \odot P))$.
:::
