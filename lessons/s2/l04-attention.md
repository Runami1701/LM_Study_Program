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
Let every position gather information from the others: queries, keys and values `(B, T, d_h)`, scores $QK^\top / \sqrt{d_h}$ `(B, T, T)`, a softmax over keys, and the weighted sum $PV$ `(B, T, d_h)`. Derive why the scale is $\sqrt{d_h}$ and the softmax's backward pass.
:::

::: notation
| Symbol | Meaning |
| --- | --- |
| $x$ | input, $(B, T, d)$ |
| $W_q, W_k, W_v$ | projection matrices, $(d, d_h)$ each |
| $Q, K, V$ | queries $xW_q$, keys $xW_k$, values $xW_v$: $(B, T, d_h)$ each |
| $d_h$ | head size (the length of each query/key/value vector) |
| $S$ | scores $QK^\top/\sqrt{d_h}$, $(B, T, T)$; $S_{ij}$ = query $i$ against key $j$ |
| $P$ | attention weights $\mathrm{softmax}(S)$ over the last axis, $(B, T, T)$ |
| $q, k$ | one query and one key vector, length $d_h$ |
| $q_m, k_m$ | their $m$-th entries |
| $i, j, m$ | query position, key position, vector entry |
| $\mathrm{Var}$, $\mathbb{E}$ | variance, expected value |
| $dP, dS$ | gradients of the loss with respect to $P$ and $S$ |
| $\delta_{ij}$ | 1 if $i = j$, else 0 |
| $\odot$ | elementwise product |
| $\top$ | transpose of the last two axes |
| $p, s$ | one row of $P$ and of $S$ |
| $L$ | the loss |
| $e$ | Euler's number |
:::

::: concept
### The idea in one sentence

Each position $i$ asks a question (its **query** $q_i$), every position $j$ advertises what it contains (its **key** $k_j$), and position $i$ collects the **values** $v_j$ of the positions whose keys match its query best.

### Step by step, with shapes

| step | formula | shape |
| --- | --- | --- |
| project | $Q = xW_q$, $K = xW_k$, $V = xW_v$ | $(B, T, d_h)$ each |
| compare | $S = QK^\top / \sqrt{d_h}$ | $(B, T, d_h)(B, d_h, T) = (B, T, T)$ |
| normalize | $P = \mathrm{softmax}(S)$ along the last (key) axis | $(B, T, T)$, rows sum to 1 |
| mix | $\text{out} = PV$ | $(B, T, T)(B, T, d_h) = (B, T, d_h)$ |

Row $i$ of $P$ is a probability distribution over positions; output $i$ is the average of the value vectors under it. $K^\top$ means transposing the last two axes of $K$: `transpose(k, (0, 2, 1))`.

### Why divide by $\sqrt{d_h}$

Take one query and one key with independent entries, each of mean 0 and variance 1 (roughly what LayerNorm'd inputs and small random projections give). Their dot product is a sum of $d_h$ terms:

$$
q \cdot k = \sum_{m=1}^{d_h} q_m k_m
$$

Each term has mean $\mathbb{E}[q_m]\mathbb{E}[k_m] = 0$ and variance $\mathbb{E}[q_m^2]\mathbb{E}[k_m^2] = 1 \cdot 1 = 1$. Independent variances add:

$$
\mathrm{Var}(q \cdot k) = \sum_{m=1}^{d_h} \mathrm{Var}(q_m k_m) = d_h
$$

So raw scores have standard deviation $\sqrt{d_h}$: about 5.7 for $d_h = 32$, 16 for $d_h = 256$. Softmax of numbers that far apart is almost one-hot: one key takes all the weight, the others get nearly 0, and nearly zero gradient (the softmax is saturated, like tanh in s1-l03). Dividing by $\sqrt{d_h}$ brings the variance back to $\frac{d_h}{d_h} = 1$, independent of the head size.

### Softmax backward, derived

For one row, $p_i = \frac{e^{s_i}}{\sum_k e^{s_k}}$. Quotient rule:

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

As arrays, for every row at once:

$$
dS = P \odot \left(dP - \mathrm{rowsum}(dP \odot P)\right)
$$

(`rowsum` with `keepdims=True` so it broadcasts back.) It is cheaper and more stable than composing softmax from `exp`, `sum` and division ops, so we make it one op, like cross-entropy.

**Three-token example.** One row of scores $s = [1, 0, -1]$ gives $p = [0.665, 0.245, 0.090]$. If the upstream gradient is $dP = [1, 0, 0]$, then $\sum_i dP_i\,p_i = 0.665$ and

$$
dS = [0.665 \cdot 0.335,\ 0.245 \cdot (-0.665),\ 0.090 \cdot (-0.665)] = [0.223,\ -0.163,\ -0.060]
$$

The entries sum to 0: adding a constant to every score changes no weight (shift invariance, s1-l04).
:::

::: think
Q: With $d_h = 64$ and unit-variance entries, what is the standard deviation of a raw score, and what happens to the softmax over 128 keys?
A: $\sqrt{64} = 8$. Scores spread over roughly $\pm 16$; after the exponential the largest dominates by factors like $e^{8}$, so each row is nearly one-hot. Gradients through the other keys vanish, and the model learns slowly.
:::

::: think
Q: Why is each row of $dS$ guaranteed to sum to zero?
A: Adding the same constant to a row of scores leaves the softmax unchanged, so the loss doesn't change; the row sum of $dS$ is the rate of change in that direction, hence 0. Algebraically: $\sum_j p_j(dP_j - c) = c - c = 0$ with $c = \sum_i dP_i p_i$.
:::

::: think
Q: Write the shapes of the two matrix products for $B = 32$, $T = 128$, $d_h = 32$. Which tensor is the largest?
A: $QK^\top$: `(32, 128, 32) @ (32, 32, 128) → (32, 128, 128)`; $PV$: `(32, 128, 128) @ (32, 128, 32) → (32, 128, 32)`. The scores and weights `(32, 128, 128)` are the largest, 524K numbers each, and they grow with $T^2$.
:::

::: hint level=1
`softmax` is the stable s1-l04 softmax plus a closure applying the boxed formula. `attention_single_head` is four lines with engine ops: three projections, scores, softmax, mix.
:::

::: hint level=2
`ops.transpose(k, (0, 2, 1))` gives $K^\top$; multiply the scores by the Python float `1.0 / math.sqrt(d_h)` (or `1.0 / np.sqrt(d_h)`, which is fine here because the engine's `*` wraps it in the data's dtype). In the softmax backward: `(g * P).sum(axis=-1, keepdims=True)`.
:::

::: hint level=3
```pseudo
softmax(x):
    P ← stable softmax of x.data over the last axis
    out ← Tensor(P, parents (x,))
    backward: x.grad += P · (g − rowsum(g · P))

attention_single_head(x, Wq, Wk, Wv, causal):
    q, k, v ← x @ Wq, x @ Wk, x @ Wv                 # (B, T, d_h)
    scores ← (q @ transpose(k, (0, 2, 1))) · (1 / sqrt(d_h))   # (B, T, T)
    (s2-l05: if causal, add the mask)
    return softmax(scores) @ v                        # (B, T, d_h)
```
:::

::: attempt
File: `workspace/stage2/attention.py` — implement `softmax` and `attention_single_head` (ignore `causal` until s2-l05).
Check: `python -m course check s2-l04`
:::

::: solution
@include solutions/stage2/attention.py#softmax

- Forward: max-subtraction, exactly as in s1-l04, for any number of leading axes.
- Backward: the derived $P \odot (dP - \mathrm{rowsum}(dP \odot P))$, using the saved `P`.

@include solutions/stage2/attention.py#attention_single_head

- The three projections are engine matmuls, so their weights get gradients automatically.
- Multiplying by a Python number goes through `Tensor.__mul__`, which wraps it in the data's dtype, so float32 is preserved.
- The `causal` branch is s2-l05.
:::

::: breakit
Remove the $1/\sqrt{d_h}$ and run attention with identity projections on random unit-variance inputs with $d_h = 256$. Print the largest entry of each row of $P$. Expect values above 0.99 for most rows (one-hot attention); with the scale, typical maxima are around 0.2 to 0.5.
:::

::: recap
- Attention: queries match keys, the softmax turns matches into weights, outputs average the values.
- Scores scale like $\sqrt{d_h}$ (you derived $\mathrm{Var}(q \cdot k) = d_h$); dividing by it keeps the softmax soft.
- Softmax backward: $\frac{\partial p_i}{\partial s_j} = p_i(\delta_{ij} - p_j)$, hence $dS = P \odot (dP - \mathrm{rowsum}(dP \odot P))$.
:::
