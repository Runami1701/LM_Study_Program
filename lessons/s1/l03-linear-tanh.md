---
id: s1-l03
title: Linear layer and tanh (forward)
stage: s1
order: 3
prereqs: [s1-l02]
files: [workspace/stage1/layers.py]
functions: [linear_forward, tanh_forward]
tests: tests/s1/test_l03_linear_tanh.py
est_minutes: 35
---

::: goal
Implement the two building blocks of the MLP's hidden layer: a linear layer `out = x @ W + b` taking `(B, n)` to `(B, m)`, and an elementwise `tanh`; each returns the cache its backward pass will need in s1-l06.
:::

::: notation
| Symbol | Meaning |
| --- | --- |
| $x$ | input batch, shape $(B, n)$; row $b$ is one example |
| $W$ | weight matrix, shape $(n, m)$ |
| $b$ | bias vector, shape $(m,)$ (the index $b$ in $x_{bi}$ is a batch row; context tells them apart) |
| $y$ | output of the linear layer, shape $(B, m)$ |
| $B$ | batch size: the number of rows of $x$ |
| $n, m$ | input and output widths of a layer |
| $x_{bi}, W_{ij}, b_j, y_{bj}$ | entries: batch row $b$, input unit $i$, output unit $j$ |
| $W_1, b_1, W_2, b_2$ | parameters of a first and a second layer |
| $h$ | hidden width (or the hidden activations, shape $(B, h)$) |
| $W', b'$ | the single layer that two stacked linear layers collapse into |
| $a$ | a scalar input to tanh |
| $\tanh$ | hyperbolic tangent, defined below |
| $e$ | Euler's number $\approx 2.71828$ in $e^{a}$ (not the embedding size here) |
| `cache` | values saved in the forward pass for the backward pass |
:::

::: concept
### The linear layer

A linear (more precisely, *affine*) layer maps each input row to an output row:

$$
y = xW + b
$$

with shapes

$$
\underbrace{(B, n)}_{x} \;\cdot\; \underbrace{(n, m)}_{W} \;+\; \underbrace{(m,)}_{b} \;\longrightarrow\; \underbrace{(B, m)}_{y}
$$

The inner dimensions ($n$ and $n$) must match and disappear; the outer ones ($B$ and $m$) remain. Entry by entry:

$$
y_{bj} = \sum_{i=1}^{n} x_{bi} W_{ij} + b_j
$$

Output unit $j$ is a weighted sum of all $n$ inputs, using column $j$ of $W$ as the weights, plus its own offset $b_j$.

We use the **row-vector convention** `x @ W`: each example is a row, and the batch is the first axis. A whole batch then goes through in one matrix product, and the same $W$ serves every row.

**Worked example** ($B = 2$, $n = 2$, $m = 3$):

$$
x = \begin{bmatrix} 1 & 2 \\ 0 & -1 \end{bmatrix}, \quad
W = \begin{bmatrix} 1 & 0 & -1 \\ 2 & 1 & 0.5 \end{bmatrix}, \quad
b = \begin{bmatrix} 0.5 & -1 & 0 \end{bmatrix}
$$

Row 0 of $xW$, one column at a time:

$$
\begin{aligned}
y_{00} &= 1 \cdot 1 + 2 \cdot 2 = 5 \\
y_{01} &= 1 \cdot 0 + 2 \cdot 1 = 2 \\
y_{02} &= 1 \cdot (-1) + 2 \cdot 0.5 = 0
\end{aligned}
$$

Row 1: $0 \cdot 1 + (-1) \cdot 2 = -2$, then $0 \cdot 0 + (-1) \cdot 1 = -1$, then $0 \cdot (-1) + (-1) \cdot 0.5 = -0.5$. So

$$
xW = \begin{bmatrix} 5 & 2 & 0 \\ -2 & -1 & -0.5 \end{bmatrix}, \qquad
xW + b = \begin{bmatrix} 5.5 & 1 & 0 \\ -1.5 & -2 & -0.5 \end{bmatrix}
$$

### Adding the bias: broadcasting again

$xW$ is $(B, m)$ but $b$ is $(m,)$. NumPy aligns shapes from the right: $(m,)$ is treated as $(1, m)$, and the length-1 axis is stretched over the $B$ rows. So the same $b$ is added to every row, with no copy of $b$ made.

In the MLP of s1-l09 the first layer has $x$ `(B, 192)` (from s1-l02's flatten), $W_1$ `(192, 512)`, $b_1$ `(512,)`: each step costs $B \cdot 192 \cdot 512$ multiply-adds, about $3.1$ million for $B = 32$.

### Why we need a nonlinearity

Stack two linear layers with nothing in between, $W_1$ `(n, h)` and $W_2$ `(h, m)`:

$$
\begin{aligned}
h &= xW_1 + b_1 \\
y &= hW_2 + b_2 \\
  &= (xW_1 + b_1)W_2 + b_2 \\
  &= xW_1W_2 + b_1W_2 + b_2 \\
  &= x\,\underbrace{(W_1W_2)}_{W'} + \underbrace{(b_1W_2 + b_2)}_{b'}
\end{aligned}
$$

$W' = W_1W_2$ has shape $(n, h)(h, m) = (n, m)$ and $b' = b_1W_2 + b_2$ has shape $(m,)$: the two layers are **exactly one linear layer**. With scalars: doubling then tripling is just multiplying by 6. Depth buys nothing without a nonlinear function between the layers.

### tanh

$$
\tanh a = \frac{e^{a} - e^{-a}}{e^{a} + e^{-a}}
$$

Values, to 3 decimals:

| $a$ | 0 | 0.5 | 1 | 2 | 3 |
| --- | --- | --- | --- | --- | --- |
| $\tanh a$ | 0 | 0.462 | 0.762 | 0.964 | 0.995 |

It is odd ($\tanh(-a) = -\tanh a$), so negative inputs mirror these values. Divide the numerator and denominator by $e^{a}$ to see where it goes for large $a$:

$$
\tanh a = \frac{e^{a} - e^{-a}}{e^{a} + e^{-a}} \cdot \frac{e^{-a}}{e^{-a}} = \frac{1 - e^{-2a}}{1 + e^{-2a}}
$$

As $a \to \infty$, $e^{-2a} \to 0$, so $\tanh a \to \frac{1 - 0}{1 + 0} = 1$; by oddness, $\tanh a \to -1$ as $a \to -\infty$. The output always lies in $(-1, 1)$.

Beyond $|a| \approx 3$ the curve is nearly flat: the unit is **saturated**, and a small change in $a$ hardly changes the output. In s1-l06 you will derive the slope and see that saturated units pass almost no gradient back. That is why s1-l09 checks what fraction of hidden units sit near $\pm 1$.

**Numerical trap.** Computed straight from the definition, $a = 1000$ gives $e^{1000}$, which overflows to `inf` (float32 overflows past $e^{88.7}$), and $\frac{\infty}{\infty}$ is `nan`. `np.tanh` is implemented stably and returns exactly 1.

### What to cache

The forward pass saves what the backward pass (s1-l06) will need:

- linear: `(x, W, b)`. The gradient for $W$ needs $x$, and the gradient for $x$ needs $W$.
- tanh: its **output**. Its derivative turns out to be $1 - \tanh^2 a$, which is written in terms of the output alone.
:::

::: think
Q: Why cache tanh's output rather than its input?
A: Its derivative is $1 - \tanh^2 a$, a function of the output. Caching the output means the backward pass needs no second call to `tanh`; caching the input would force exactly that recomputation.
:::

::: think
Q: In $y = xW + b$ with $x$ `(32, 192)` and $W$ `(192, 512)`, what is $b$'s shape, how many numbers does it hold, and how many additions does adding it cost?
A: $(512,)$: one offset per output unit, 512 numbers, shared by all 32 rows. Adding it costs $32 \times 512 = 16{,}384$ additions, but $b$ itself is never copied: broadcasting reuses the same 512 numbers for every row.
:::

::: think
Q: A model has layers 192 → 512 → 65 with **no** tanh in between. How many parameters does it have, and how many would a single 192 → 65 layer need? Which can represent more functions?
A: Two layers: $192 \cdot 512 + 512 + 512 \cdot 65 + 65 = 98{,}304 + 512 + 33{,}280 + 65 = 132{,}161$. One layer: $192 \cdot 65 + 65 = 12{,}545$. Both represent exactly the same set of functions, because the pair collapses to $xW' + b'$. Ten times the parameters, zero extra power.
:::

::: hint level=1
The linear layer is one matrix product plus a bias, and the bias needs no special handling. NumPy already has a stable tanh. Each function returns two things: the result and its cache.
:::

::: hint level=2
The `@` operator multiplies `(B, n)` by `(n, m)`; adding a `(m,)` array to a `(B, m)` array broadcasts it over the rows. Use `np.tanh` (it keeps float32 and never overflows). The linear cache is the tuple `(x, W, b)`; the tanh cache is the output array itself.
:::

::: hint level=3
```pseudo
linear_forward(x, W, b):
    out ← x matrix-times W, plus b broadcast over rows     # (B, n)(n, m) + (m,) → (B, m)
    return out, (x, W, b)

tanh_forward(x):
    out ← elementwise tanh of x                           # same shape as x
    return out, out
```
:::

::: attempt
File: `workspace/stage1/layers.py` — implement `linear_forward` and `tanh_forward`.
Check: `python -m course check s1-l03`
:::

::: solution
@include solutions/stage1/layers.py#linear_forward

- `x @ W` is the batched matrix product, `(B, n) @ (n, m) → (B, m)`, computed by BLAS.
- `+ b` broadcasts the `(m,)` bias over the $B$ rows, so one bias vector serves the whole batch.
- All three inputs are float32, so the output stays float32: nothing here introduces a float64.
- The cache `(x, W, b)` stores references, not copies, so caching is free.

@include solutions/stage1/layers.py#tanh_forward

- `np.tanh` is elementwise, works on any shape and keeps the dtype.
- It is numerically stable for huge inputs (it returns exactly ±1), unlike a hand-written $\frac{e^{a} - e^{-a}}{e^{a} + e^{-a}}$.
- The output doubles as the cache, because the backward pass needs only $\tanh a$.
:::

::: breakit
**The collapse, numerically.** With random float32 `x` `(4, 3)`, `W1` `(3, 5)`, `b1` `(5,)`, `W2` `(5, 2)`, `b2` `(2,)`, compute two stacked linear layers and then the single layer with `W1 @ W2` and `b1 @ W2 + b2`. Expect `np.allclose` to say they are equal (differences around $10^{-7}$, float32 rounding). Insert `np.tanh` between the layers and the equality breaks. In s1-l10 you will remove tanh from the real model and watch the training loss stall well above the tanh model's.
:::

::: breakit
**Overflow.** Write tanh yourself as `(np.exp(a) - np.exp(-a)) / (np.exp(a) + np.exp(-a))` and apply it to `np.array([1000.0], dtype=np.float32)`. Expect a `RuntimeWarning: overflow` and the result `nan`. Then try the rewritten form $\frac{1 - e^{-2a}}{1 + e^{-2a}}$ for $a = 1000$: it returns 1.0 (but now overflows for $a = -1000$; `np.tanh` handles both signs).
:::

::: recap
- $y = xW + b$: `(B, n)(n, m) + (m,) → (B, m)`, bias broadcast over rows, worked by hand.
- Two linear layers without a nonlinearity collapse into one; you derived it.
- tanh maps into $(-1, 1)$, saturates beyond $|a| \approx 3$, and must be computed stably.
- Each layer returns a cache for its backward pass in s1-l06.
:::
