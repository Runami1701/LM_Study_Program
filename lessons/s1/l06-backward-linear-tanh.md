---
id: s1-l06
title: Backward through linear and tanh
stage: s1
order: 6
prereqs: [s1-l05]
files: [workspace/stage1/layers.py]
functions: [linear_backward, tanh_backward]
tests: tests/s1/test_l06_backward.py
est_minutes: 50
---

::: goal
Given the upstream gradient `dout` `(B, m)` of a linear layer, compute `dx` `(B, n)`, `dW` `(n, m)` and `db` `(m,)`; and push a gradient back through tanh. Derive each formula from the definition with index notation.
:::

::: notation
| Symbol | Meaning |
| --- | --- |
| $x, W, b, y$ | input `(B, n)`, weights `(n, m)`, bias `(m,)`, output `(B, m)`, with $y = xW + b$ |
| $x_{bi}, W_{ij}, b_j, y_{bj}$ | entries; $b$ indexes the batch row, $i$ an input unit, $j$ an output unit |
| $B, n, m$ | batch size, input width, output width |
| $L$ | the scalar loss at the end of the network |
| $\frac{\partial L}{\partial y_{bj}}$ | the upstream gradient: given to us as `dout` |
| $d$ | shorthand: $dW$ means the array of $\frac{\partial L}{\partial W_{ij}}$, same shape as $W$ |
| $\top$ | transpose: $(x^\top)_{ib} = x_{bi}$ |
| $a$ | the input to tanh; $t = \tanh a$ its output |
| $t$ | tanh's output (what the cache holds) |
| $e$ | Euler's number, as in $e^{a}$ |
| $g$ | an upstream gradient arriving at a function |
| $\theta$ | any quantity we differentiate with respect to (a weight, an input) |
| $o$ | one output that $\theta$ influences |
| $u, v$ | numerator $e^{a} - e^{-a}$ and denominator $e^{a} + e^{-a}$ of tanh |
:::

::: concept
### The chain rule as bookkeeping

Backpropagation runs the network in reverse. Each layer receives $\frac{\partial L}{\partial \text{output}}$ from above (`dout`) and must return $\frac{\partial L}{\partial \text{input}}$ and $\frac{\partial L}{\partial \text{parameters}}$. The rule for every case: **if a quantity influences the loss through several outputs, add up the contributions**:

$$
\frac{\partial L}{\partial \theta} = \sum_{\text{outputs } o} \frac{\partial L}{\partial o} \cdot \frac{\partial o}{\partial \theta}
$$

upstream gradient times local derivative, summed over every path.

### dW from index notation

$y_{bj} = \sum_{i} x_{bi} W_{ij} + b_j$. Which outputs does one weight $W_{ij}$ touch? It appears in $y_{bj}$ for **every** row $b$ (same column $j$), with local derivative $\frac{\partial y_{bj}}{\partial W_{ij}} = x_{bi}$. Summing over those paths:

$$
\begin{aligned}
\frac{\partial L}{\partial W_{ij}}
&= \sum_{b} \frac{\partial L}{\partial y_{bj}} \, x_{bi} \\
&= \sum_{b} (x^\top)_{ib} \, \text{dout}_{bj} \\
&= (x^\top \, \text{dout})_{ij}
\end{aligned}
$$

So $dW = x^\top \, \text{dout}$, with shapes $(n, B)(B, m) = (n, m)$: the shape of $W$, as it must be.

### dx

Input $x_{bi}$ appears in $y_{bj}$ for every output column $j$ of the **same** row $b$, with local derivative $W_{ij}$:

$$
\begin{aligned}
\frac{\partial L}{\partial x_{bi}}
&= \sum_{j} \text{dout}_{bj} \, W_{ij} \\
&= \sum_{j} \text{dout}_{bj} \, (W^\top)_{ji} \\
&= (\text{dout} \, W^\top)_{bi}
\end{aligned}
$$

So $dx = \text{dout} \, W^\top$: $(B, m)(m, n) = (B, n)$.

### db

$b_j$ appears in $y_{bj}$ for every row $b$, with local derivative 1:

$$
\frac{\partial L}{\partial b_j} = \sum_b \text{dout}_{bj}
$$

So $db$ is `dout` summed over the batch axis: shape $(m,)$. This is the backward side of broadcasting: the forward pass copied $b$ to all $B$ rows, so the backward pass adds the $B$ gradients back together.

### Worked example with numbers ($B = n = m = 2$)

$$
x = \begin{bmatrix} 1 & 2 \\ 3 & 4 \end{bmatrix}, \quad
W = \begin{bmatrix} 1 & 0 \\ -1 & 2 \end{bmatrix}, \quad
\text{dout} = \begin{bmatrix} 1 & 0 \\ 0 & 1 \end{bmatrix}
$$

Take $W_{00}$: it feeds $y_{00}$ (with $x_{00} = 1$) and $y_{10}$ (with $x_{10} = 3$), and $\text{dout}_{00} = 1$, $\text{dout}_{10} = 0$. So $\frac{\partial L}{\partial W_{00}} = 1 \cdot 1 + 0 \cdot 3 = 1$. Doing all four entries is the same as

$$
dW = x^\top \text{dout} = \begin{bmatrix} 1 & 3 \\ 2 & 4 \end{bmatrix}, \quad
dx = \text{dout}\, W^\top = \begin{bmatrix} 1 & -1 \\ 0 & 2 \end{bmatrix}, \quad
db = [1,\ 1]
$$

### The shape-matching trick

Once derived, you can **recover** these formulas from shapes alone, which is a great sanity check. $dW$ must be $(n, m)$ and must combine $x$ $(B, n)$ with dout $(B, m)$. The only product of those two that gives $(n, m)$ is $x^\top \text{dout}$: $(n, B)(B, m)$. Likewise $dx$ must be $(B, n)$ from dout $(B, m)$ and $W$ $(n, m)$: only $\text{dout}\,W^\top$ fits. A gradient always has the shape of the thing it is the gradient of. We will use this trick for every layer from now on.

### tanh

Write $t = \tanh a = \frac{u}{v}$ with $u = e^{a} - e^{-a}$ and $v = e^{a} + e^{-a}$. Then $u' = e^{a} + e^{-a} = v$ and $v' = e^{a} - e^{-a} = u$. The quotient rule $\left(\frac{u}{v}\right)' = \frac{u'v - uv'}{v^2}$ gives

$$
\begin{aligned}
\frac{d}{da}\tanh a &= \frac{v \cdot v - u \cdot u}{v^2} \\
&= 1 - \frac{u^2}{v^2} \\
&= 1 - \tanh^2 a = 1 - t^2
\end{aligned}
$$

It is written in terms of the output $t$, which is why the forward pass cached `out`. tanh is elementwise, so each input affects only its own output, and the chain rule has a single path:

$$
\frac{\partial L}{\partial a} = g \cdot (1 - t^2) \quad \text{(elementwise, } g = \text{dout)}
$$

Numbers: at $a = 1$, $1 - \tanh^2 1 = 0.420$; at $a = 3$, $1 - 0.995^2 = 0.0099$. A saturated unit passes back about 1% of its gradient: the "vanishing gradient" of s1-l03.
:::

::: think
Q: Why must `db` be a sum over the batch, and not, say, the first row of `dout`?
A: The same bias $b_j$ was added to every row, so it influenced all $B$ outputs in column $j$. Each of those paths contributes to its gradient, and contributions from several paths add (the chain rule). Taking one row ignores $B - 1$ of them.
:::

::: think
Q: Use the shape-matching trick: in s1-l09 you will backpropagate through `logits = h @ W2 + b2` with $h$ `(B, 512)` and $W_2$ `(512, 65)`. What are the shapes and formulas of `dh` and `dW2`?
A: `dlogits` is `(B, 65)`. `dW2` must be `(512, 65)` = $h^\top$ `(512, B)` times `dlogits` `(B, 65)`. `dh` must be `(B, 512)` = `dlogits` `(B, 65)` times $W_2^\top$ `(65, 512)`.
:::

::: think
Q: A tanh unit has $a = 4$. Roughly what fraction of the incoming gradient gets through, and why does this matter at initialization?
A: $1 - \tanh^2 4 \approx 1 - 0.9993^2 \approx 0.0013$, about 0.1%. If initialization makes many pre-activations large, most hidden units pass almost no gradient and those weights barely learn. That's why s1-l09 scales $W_1$ to keep pre-activations around unit size.
:::

::: hint level=1
Each formula is one line once you have the cache. Two are matrix products and one is a sum; tanh's backward is elementwise.
:::

::: hint level=2
`.T` transposes; `@` multiplies. Sum over the batch with `.sum(axis=0)`. For tanh, the cache is the output $t$, so the local derivative is $1 - t^2$, multiplied elementwise by `dout`.
:::

::: hint level=3
```pseudo
linear_backward(dout, cache):
    x, W, b ← cache
    dx ← dout times W transposed      # (B, m)(m, n) → (B, n)
    dW ← x transposed times dout      # (n, B)(B, m) → (n, m)
    db ← dout summed over rows        # (m,)
    return dx, dW, db

tanh_backward(dout, cache):
    t ← cache
    return dout · (1 − t²)            # elementwise
```
:::

::: attempt
File: `workspace/stage1/layers.py` — implement `linear_backward` and `tanh_backward`.
Check: `python -m course check s1-l06`
:::

::: solution
@include solutions/stage1/layers.py#linear_backward

- `dout @ W.T`: $dx_{bi} = \sum_j \text{dout}_{bj} W_{ij}$, a sum over output units.
- `x.T @ dout`: $dW_{ij} = \sum_b x_{bi}\, \text{dout}_{bj}$, a sum over the batch. Larger batches add more terms, which is why the loss is a mean (s1-l05) rather than a sum.
- `dout.sum(axis=0)` undoes the broadcast of `b` over rows.
- `b` is unpacked but unused: the cache format stays `(x, W, b)` so it mirrors the forward pass.

@include solutions/stage1/layers.py#tanh_backward

- Elementwise product: each pre-activation affects only its own output.
- `1.0 - cache * cache` keeps float32 (`1.0` is a weak Python float).
:::

::: breakit
Compute `db` as `dout` itself, without the sum, and pass it to a parameter update `b -= lr * db`. Expect `ValueError: non-broadcastable output operand with shape (m,) doesn't match the broadcast shape (B, m)`: NumPy refuses to write a `(B, m)` result into the `(m,)` array. The shape mismatch is the bug telling you that $B$ paths were not summed.
:::

::: recap
- $dW = x^\top \text{dout}$, $dx = \text{dout}\,W^\top$, $db = \sum_b \text{dout}_b$, derived entry by entry and checked with numbers.
- The shape-matching trick recovers these from shapes alone; a gradient has its variable's shape.
- $\frac{d}{da}\tanh a = 1 - \tanh^2 a$: cheap from the cached output, and tiny for saturated units.
:::
