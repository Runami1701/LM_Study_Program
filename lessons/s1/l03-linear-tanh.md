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
You will write the two parts of the model's hidden layer: a linear layer `out = x @ W + b`, which turns `(B, n)` into `(B, m)`, and the function tanh, applied to every entry. Each function also returns a **cache**: the values its backward pass will need in s1-l06.
:::

::: notation
| Symbol | Meaning |
| --- | --- |
| $x$ | input batch, shape $(B, n)$; row $b$ is one example |
| $W$ | weight matrix, shape $(n, m)$ |
| $b$ | bias vector, shape $(m,)$ (in $x_{bi}$, $b$ is a row number; the context tells them apart) |
| $y$ | output of the linear layer, shape $(B, m)$ |
| $B$ | batch size: the number of rows of $x$ |
| $n, m$ | input width and output width of a layer |
| $x_{bi}, W_{ij}, b_j, y_{bj}$ | entries: batch row $b$, input $i$, output $j$ |
| $W_1, b_1, W_2, b_2$ | weights and biases of a first and a second layer |
| $h$ | width of the hidden layer (or its output, shape $(B, h)$) |
| $W', b'$ | the single layer that two stacked linear layers turn into |
| $a$ | a number that goes into tanh |
| $\tanh$ | hyperbolic tangent, defined below |
| $e$ | Euler's number $\approx 2.71828$, as in $e^{a}$ (not the embedding size here) |
| `cache` | values saved in the forward pass for the backward pass |
:::

::: numpy
### Matrix product plus a bias (n-l01, n-l04)

`x @ W` is the matrix product. Adding a vector of shape `(m,)` to a `(B, m)` matrix adds it to every row (broadcasting):

```python
x = np.array([[1., 2.], [0., -1.]])               # (2, 2)
W = np.array([[1., 0., -1.], [2., 1., 0.5]])      # (2, 3)
b = np.array([0.5, -1., 0.])                      # (3,)
print(x @ W + b)
# [[ 5.5  1.   0. ]
#  [-1.5 -2.  -0.5]]
```

### tanh of every entry

`np.tanh` applies tanh to every entry and keeps the shape and dtype. It never overflows:

```python
print(np.tanh(np.array([-1000., 0., 1000.])))     # [-1.  0.  1.]
```

### Returning two things

A Python function can return two values as a tuple: `return out, cache`. The caller unpacks them with `out, cache = linear_forward(x, W, b)`.
:::

::: concept
### The linear layer

A linear layer (strictly, an *affine* layer) maps each input row to an output row:

$$
y = xW + b
$$

with shapes

$$
\underbrace{(B, n)}_{x} \;\cdot\; \underbrace{(n, m)}_{W} \;+\; \underbrace{(m,)}_{b} \;\longrightarrow\; \underbrace{(B, m)}_{y}
$$

The inner sizes ($n$ and $n$) must match and disappear; the outer sizes ($B$ and $m$) remain. Entry by entry:

$$
y_{bj} = \sum_{i=1}^{n} x_{bi} W_{ij} + b_j
$$

Output $j$ is a weighted sum of all $n$ inputs, with column $j$ of $W$ as the weights, plus its own offset $b_j$.

The course puts each example in a **row** and writes `x @ W`. Then one matrix product handles the whole batch, and the same $W$ serves every row.

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

$xW$ has shape $(B, m)$ and $b$ has shape $(m,)$. Broadcasting adds the same $b$ to every row, without copying $b$.

In the model of s1-l09 the first layer has $x$ `(B, 192)`, $W_1$ `(192, 512)` and $b_1$ `(512,)`. One step costs $B \cdot 192 \cdot 512$ multiply-adds, about 3.1 million for $B = 32$.

### Two linear layers are only one

Stack two linear layers, $W_1$ `(n, h)` and $W_2$ `(h, m)`, with nothing between them:

$$
\begin{aligned}
h &= xW_1 + b_1 \\
y &= hW_2 + b_2 \\
  &= (xW_1 + b_1)W_2 + b_2 \\
  &= xW_1W_2 + b_1W_2 + b_2 \\
  &= x\,\underbrace{(W_1W_2)}_{W'} + \underbrace{(b_1W_2 + b_2)}_{b'}
\end{aligned}
$$

$W' = W_1W_2$ has shape $(n, m)$ and $b' = b_1W_2 + b_2$ has shape $(m,)$. So the two layers compute **exactly what one linear layer computes**. With plain numbers: doubling and then tripling is the same as multiplying by 6. Extra layers only help if a nonlinear function sits between them.

### tanh

$$
\tanh a = \frac{e^{a} - e^{-a}}{e^{a} + e^{-a}}
$$

Values, to 3 decimals:

| $a$ | 0 | 0.5 | 1 | 2 | 3 |
| --- | --- | --- | --- | --- | --- |
| $\tanh a$ | 0 | 0.462 | 0.762 | 0.964 | 0.995 |

tanh is odd: $\tanh(-a) = -\tanh a$. To see where it goes for large $a$, divide the top and bottom by $e^{a}$:

$$
\tanh a = \frac{e^{a} - e^{-a}}{e^{a} + e^{-a}} \cdot \frac{e^{-a}}{e^{-a}} = \frac{1 - e^{-2a}}{1 + e^{-2a}}
$$

As $a \to \infty$, $e^{-2a} \to 0$, so $\tanh a \to 1$. For $a \to -\infty$ it goes to $-1$. The output always lies between $-1$ and $1$.

For $|a| > 3$ the curve is almost flat: the unit is **saturated**, and a small change in $a$ hardly changes the output. In s1-l06 you will see that a saturated unit passes almost no gradient back. That is why s1-l09 checks how many hidden units sit near $\pm 1$.

**A trap.** Computed straight from the definition, $a = 1000$ needs $e^{1000}$. That is `inf` in float32 (anything above $e^{88.7}$ is), and $\frac{\infty}{\infty}$ is `nan`. `np.tanh` avoids this and returns exactly 1.

### What to save for the backward pass

- linear: save `(x, W, b)`. The gradient of $W$ needs $x$, and the gradient of $x$ needs $W$.
- tanh: save its **output**. Its derivative is $1 - \tanh^2 a$, which only needs the output.
:::

::: think
Q: Why save tanh's output rather than its input?
A: The derivative is $1 - \tanh^2 a$, which uses the output. With the output saved, the backward pass needs no second call to `tanh`.
:::

::: think
Q: In $y = xW + b$ with $x$ `(32, 192)` and $W$ `(192, 512)`, what is $b$'s shape, and how many additions does adding it cost?
A: $(512,)$: one offset per output, shared by all 32 rows. Adding it costs $32 \times 512 = 16{,}384$ additions. Broadcasting uses the same 512 numbers for every row without copying them.
:::

::: think
Q: A model has layers 192 → 512 → 65 and **no** tanh. How many parameters does it have, and how many does a single 192 → 65 layer have? Which can compute more?
A: Two layers: $192 \cdot 512 + 512 + 512 \cdot 65 + 65 = 132{,}161$. One layer: $192 \cdot 65 + 65 = 12{,}545$. Both compute exactly the same set of functions, because the pair collapses into $xW' + b'$. Ten times the parameters buy nothing.
:::

::: hint level=1
The linear layer is one matrix product plus the bias; the bias needs no special handling. NumPy already has a safe tanh. Each function returns two things: the result and its cache.
:::

::: hint level=2
`x @ W + b` (broadcasting adds `b` to every row). `np.tanh(x)`. The linear cache is the tuple `(x, W, b)`; the tanh cache is the output array.
:::

::: hint level=3
```pseudo
linear_forward(x, W, b):
    out ← x matrix-times W, plus b on every row      # (B, n)(n, m) + (m,) → (B, m)
    return out, (x, W, b)

tanh_forward(x):
    out ← tanh of every entry of x                   # same shape as x
    return out, out
```
:::

::: attempt
File: `workspace/stage1/layers.py` — fill in `linear_forward` and `tanh_forward`.
Check: `python -m course check s1-l03`
:::

::: solution
@include solutions/stage1/layers.py#linear_forward

- `x @ W` is the matrix product for the whole batch.
- `+ b` adds the `(m,)` bias to all $B$ rows.
- All three inputs are float32, so the output stays float32.
- The cache stores references to the arrays, not copies, so it costs nothing.

@include solutions/stage1/layers.py#tanh_forward

- `np.tanh` works on any shape and keeps the dtype.
- It returns exactly ±1 for huge inputs, unlike a hand-written $\frac{e^{a} - e^{-a}}{e^{a} + e^{-a}}$.
- The output doubles as the cache, because the backward pass needs only $\tanh a$.
:::

::: breakit
**Two layers collapse into one.** With random float32 `x` `(4, 3)`, `W1` `(3, 5)`, `b1` `(5,)`, `W2` `(5, 2)` and `b2` `(2,)`, compute two stacked linear layers. Then compute one layer with `W1 @ W2` and `b1 @ W2 + b2`. `np.allclose` reports the results as equal (they differ by about $10^{-7}$, float32 rounding). Put `np.tanh` between the two layers and they are no longer equal.
:::

::: breakit
**Overflow.** Write tanh yourself as `(np.exp(a) - np.exp(-a)) / (np.exp(a) + np.exp(-a))` and apply it to `np.array([1000.0], dtype=np.float32)`. You get `RuntimeWarning: overflow` and the result `nan`.
:::

::: recap
- $y = xW + b$: `(B, n)(n, m) + (m,) → (B, m)`, with the bias added to every row.
- Two linear layers without a nonlinear function in between are one linear layer; you derived it.
- tanh maps every number into $(-1, 1)$, is flat beyond $|a| \approx 3$, and must be computed safely.
- Each layer returns a cache for its backward pass.
:::
