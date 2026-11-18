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
A linear layer receives `dout` `(B, m)`, the gradient of the loss with respect to its output. You will compute the gradients with respect to its input and parameters: `dx` `(B, n)`, `dW` `(n, m)` and `db` `(m,)`. You will also pass a gradient back through tanh. Each formula is derived from the definition, one entry at a time.
:::

::: notation
| Symbol | Meaning |
| --- | --- |
| $x, W, b, y$ | input `(B, n)`, weights `(n, m)`, bias `(m,)`, output `(B, m)`, with $y = xW + b$ |
| $x_{bi}, W_{ij}, b_j, y_{bj}$ | entries; $b$ is a batch row, $i$ an input, $j$ an output |
| $B, n, m$ | batch size, input width, output width |
| $L$ | the loss at the end of the network |
| $\frac{\partial L}{\partial y_{bj}}$ | the gradient arriving from above, given to us as `dout` |
| $d$ | shorthand: $dW$ is the array of all $\frac{\partial L}{\partial W_{ij}}$, same shape as $W$ |
| $\top$ | transpose: $(x^\top)_{ib} = x_{bi}$ |
| $a$ | the input of tanh; $t = \tanh a$ is its output |
| $t$ | tanh's output (what the cache holds) |
| $e$ | Euler's number, as in $e^{a}$ |
| $g$ | the gradient arriving at a function from above |
| $\theta$ | any quantity we differentiate with respect to (a weight, an input) |
| $o$ | one output that $\theta$ affects |
| $u, v$ | numerator $e^{a} - e^{-a}$ and denominator $e^{a} + e^{-a}$ of tanh |
:::

::: numpy
### Transpose, sum over the batch, multiply entry by entry

```python
x = np.array([[1., 2.],
              [3., 4.]])
print(x.T)                 # rows become columns
# [[1. 3.]
#  [2. 4.]]
print(x.sum(axis=0))       # [4. 6.]   one sum per column: adds the rows together
print(x * np.array([[10., 0.], [0., 1.]]))   # entry by entry, not a matrix product
# [[10.  0.]
#  [ 0.  4.]]
```
:::

::: concept
### The chain rule as bookkeeping

Backpropagation runs the network backwards. Each layer receives $\frac{\partial L}{\partial \text{output}}$ from the layer above (`dout`). It must return $\frac{\partial L}{\partial \text{input}}$ for the layer below, and $\frac{\partial L}{\partial \text{parameter}}$ for each of its own parameters. One rule covers every case: **if a quantity affects the loss through several outputs, add up the contributions**:

$$
\frac{\partial L}{\partial \theta} = \sum_{\text{outputs } o} \frac{\partial L}{\partial o} \cdot \frac{\partial o}{\partial \theta}
$$

In words: for each path, multiply the gradient from above by the local derivative, then add over all paths.

### dW, one entry at a time

$y_{bj} = \sum_{i} x_{bi} W_{ij} + b_j$. Which outputs does one weight $W_{ij}$ affect? It appears in $y_{bj}$ for **every** row $b$ (always column $j$), with local derivative $\frac{\partial y_{bj}}{\partial W_{ij}} = x_{bi}$. Add up those paths:

$$
\begin{aligned}
\frac{\partial L}{\partial W_{ij}}
&= \sum_{b} \frac{\partial L}{\partial y_{bj}} \, x_{bi} \\
&= \sum_{b} (x^\top)_{ib} \, \text{dout}_{bj} \\
&= (x^\top \, \text{dout})_{ij}
\end{aligned}
$$

So $dW = x^\top \, \text{dout}$. Shapes: $(n, B)(B, m) = (n, m)$, the shape of $W$, as it must be.

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

So $dx = \text{dout} \, W^\top$. Shapes: $(B, m)(m, n) = (B, n)$.

### db

$b_j$ appears in $y_{bj}$ for every row $b$, with local derivative 1:

$$
\frac{\partial L}{\partial b_j} = \sum_b \text{dout}_{bj}
$$

So $db$ is `dout` added up over the batch axis, shape $(m,)$. This is the backward side of broadcasting: the forward pass added the same $b$ to all $B$ rows, so the backward pass adds the $B$ gradients together.

### Worked example ($B = n = m = 2$)

$$
x = \begin{bmatrix} 1 & 2 \\ 3 & 4 \end{bmatrix}, \quad
W = \begin{bmatrix} 1 & 0 \\ -1 & 2 \end{bmatrix}, \quad
\text{dout} = \begin{bmatrix} 1 & 0 \\ 0 & 1 \end{bmatrix}
$$

Take $W_{00}$. It feeds $y_{00}$ (with $x_{00} = 1$) and $y_{10}$ (with $x_{10} = 3$), and the gradients there are $\text{dout}_{00} = 1$ and $\text{dout}_{10} = 0$. So $\frac{\partial L}{\partial W_{00}} = 1 \cdot 1 + 0 \cdot 3 = 1$. All four entries at once:

$$
dW = x^\top \text{dout} = \begin{bmatrix} 1 & 3 \\ 2 & 4 \end{bmatrix}, \quad
dx = \text{dout}\, W^\top = \begin{bmatrix} 1 & -1 \\ 0 & 2 \end{bmatrix}, \quad
db = [1,\ 1]
$$

### The shape-matching trick

Once you have derived these formulas, you can rebuild them from shapes alone. This is a great check. $dW$ must have shape $(n, m)$ and must combine $x$ $(B, n)$ with dout $(B, m)$. The only product of the two that gives $(n, m)$ is $x^\top \text{dout}$: $(n, B)(B, m)$. Likewise, $dx$ must be $(B, n)$, made from dout $(B, m)$ and $W$ $(n, m)$; only $\text{dout}\,W^\top$ fits. **A gradient always has the shape of the thing it belongs to.** Every later layer uses this trick.

### tanh

Write $t = \tanh a = \frac{u}{v}$ with $u = e^{a} - e^{-a}$ and $v = e^{a} + e^{-a}$. Then $u' = e^{a} + e^{-a} = v$ and $v' = e^{a} - e^{-a} = u$. The quotient rule $\left(\frac{u}{v}\right)' = \frac{u'v - uv'}{v^2}$ gives

$$
\begin{aligned}
\frac{d}{da}\tanh a &= \frac{v \cdot v - u \cdot u}{v^2} \\
&= 1 - \frac{u^2}{v^2} \\
&= 1 - \tanh^2 a = 1 - t^2
\end{aligned}
$$

The result uses only the output $t$; that is why the forward pass saved `out`. tanh works on each entry separately, so each input affects only its own output, and there is a single path:

$$
\frac{\partial L}{\partial a} = g \cdot (1 - t^2) \quad \text{(entry by entry, with } g = \text{dout)}
$$

Numbers: at $a = 1$, $1 - \tanh^2 1 = 0.420$; at $a = 3$, $1 - 0.995^2 = 0.0099$. A saturated unit passes back only about 1% of its gradient. This is the "vanishing gradient" mentioned in s1-l03.
:::

::: think
Q: Why must `db` add up `dout` over the batch, and not just take its first row?
A: The same bias $b_j$ was added to every row, so it affected all $B$ outputs in column $j$. Each of those paths contributes to its gradient, and the chain rule adds them. Taking one row ignores $B - 1$ of them.
:::

::: think
Q: Use the shape-matching trick. In s1-l09 you will go backwards through `logits = h @ W2 + b2`, with $h$ `(B, 512)` and $W_2$ `(512, 65)`. What are the shapes and formulas of `dh` and `dW2`?
A: `dlogits` is `(B, 65)`. `dW2` must be `(512, 65)`: $h^\top$ `(512, B)` times `dlogits` `(B, 65)`. `dh` must be `(B, 512)`: `dlogits` `(B, 65)` times $W_2^\top$ `(65, 512)`.
:::

::: think
Q: A tanh unit has $a = 4$. Roughly what fraction of the incoming gradient gets through, and why does this matter when the model starts?
A: $1 - \tanh^2 4 \approx 0.0013$, about 0.1%. If the starting weights make many inputs of tanh large, most hidden units pass almost no gradient, and their weights barely learn. s1-l09 scales $W_1$ to prevent this.
:::

::: hint level=1
Each formula is one line once you unpack the cache. Two are matrix products and one is a sum. tanh's backward works entry by entry.
:::

::: hint level=2
`.T` transposes and `@` multiplies matrices. Add over the batch with `.sum(axis=0)`. For tanh the cache is the output $t$, so the local derivative is `1 - t * t`; multiply it entry by entry with `dout`.
:::

::: hint level=3
```pseudo
linear_backward(dout, cache):
    x, W, b ← cache
    dx ← dout times W transposed      # (B, m)(m, n) → (B, n)
    dW ← x transposed times dout      # (n, B)(B, m) → (n, m)
    db ← dout added up over the rows  # (m,)
    return dx, dW, db

tanh_backward(dout, cache):
    t ← cache
    return dout · (1 − t²)            # entry by entry
```
:::

::: attempt
The editor below opens `workspace/stage1/layers.py`: fill in `linear_backward` and `tanh_backward`.
Check: press **Run check** in the Check section below.
:::

::: solution
@include solutions/stage1/layers.py#linear_backward

- `dout @ W.T` computes $dx_{bi} = \sum_j \text{dout}_{bj} W_{ij}$.
- `x.T @ dout` computes $dW_{ij} = \sum_b x_{bi}\, \text{dout}_{bj}$: a sum over the batch.
- `dout.sum(axis=0)` undoes the broadcast of `b` over the rows.
- `b` is unpacked but not used; the cache keeps the form `(x, W, b)` to mirror the forward pass.

@include solutions/stage1/layers.py#tanh_backward

- Entry by entry: each input of tanh affects only its own output.
- `1.0 - cache * cache` keeps float32.
:::

::: breakit
Compute `db` as `dout` itself, without the sum, and apply an update `b -= lr * db`. NumPy refuses with `ValueError: non-broadcastable output operand with shape (m,) doesn't match the broadcast shape (B, m)`: it cannot write a `(B, m)` result into the `(m,)` array. The shape error tells you that the $B$ paths were never added up.
:::

::: recap
- $dW = x^\top \text{dout}$, $dx = \text{dout}\,W^\top$, $db = \sum_b \text{dout}_b$, derived entry by entry and checked with numbers.
- The shape-matching trick rebuilds these formulas from shapes; a gradient has the shape of its variable.
- $\frac{d}{da}\tanh a = 1 - \tanh^2 a$: cheap from the saved output, and tiny for saturated units.
:::
