---
id: s1-l11
title: Adam
stage: s1
order: 11
prereqs: [s1-l10]
files: [workspace/stage1/optim.py]
functions: [Adam]
tests: tests/s1/test_l11_adam.py
est_minutes: 45
---

::: goal
You will write the Adam optimizer as a class. Its `step(grads)` method updates every parameter in place, using momentum, a separate step size for each parameter, and bias correction.
:::

::: notation
| Symbol | Meaning |
| --- | --- |
| $g_t$ | the gradient at step $t$ (for one parameter entry) |
| $t$ | the step count, starting at $t = 1$ |
| $m_t$ | first moment: a running average of the gradients ("momentum") |
| $v_t$ | second moment: a running average of the squared gradients |
| $\beta_1, \beta_2$ | how slowly $m$ and $v$ forget; defaults $0.9$ and $0.999$ |
| $\hat m_t, \hat v_t$ | the averages after bias correction |
| $\eta$ | learning rate |
| $\epsilon$ | a tiny number ($10^{-8}$) that prevents division by zero |
| $\theta$ | a parameter entry |
| $g$ | a constant gradient (in the bias-correction argument) |
| $s$ | a counter in a sum |
:::

::: numpy
### Zero arrays of the same shape

```python
g = np.array([0.5, -1.0])
m = np.zeros_like(g)          # zeros with g's shape and dtype
```

### Update a running average in place

```python
m *= 0.9                      # m = 0.9 * m, in place
m += 0.1 * g                  # m = m + 0.1 * g, in place
print(m)                      # [ 0.05 -0.1 ]
```

### Powers

`0.9 ** 3` is $0.9^3$:

```python
print(0.9 ** 3)               # 0.7290000000000001
```

### A small Python class

A class bundles data with the functions that use it. `__init__` runs once when you create the object and stores data on `self`. Other methods read and change that data:

```python
class Counter:
    def __init__(self, start):
        self.n = start        # stored on the object
    def step(self):
        self.n += 1           # changes the stored value

c = Counter(5)
c.step()
print(c.n)                    # 6
```

`Adam` works the same way: `__init__` stores the parameters and creates `m` and `v`; `step` updates them.
:::

::: concept
Plain SGD has two weaknesses. Its direction is noisy, because every batch disagrees a little. And one learning rate must serve every parameter, although some get gradients of size $10^{-4}$ and others of size $1$. Adam fixes both.

### Momentum: a running average of the gradients

$$
m_t = \beta_1 m_{t-1} + (1 - \beta_1)\, g_t, \qquad m_0 = 0
$$

Write out the first three steps with $\beta_1 = 0.9$:

$$
\begin{aligned}
m_1 &= 0.1\, g_1 \\
m_2 &= 0.9 \cdot 0.1\, g_1 + 0.1\, g_2 = 0.09\, g_1 + 0.1\, g_2 \\
m_3 &= 0.081\, g_1 + 0.09\, g_2 + 0.1\, g_3
\end{aligned}
$$

Each older gradient gets 0.9 times the weight of the next newer one. So $m$ is a smooth average over roughly the last $\frac{1}{1 - \beta_1} = 10$ steps. Noise that changes sign from batch to batch averages out; a steady direction builds up.

### Second moment: how big are this parameter's gradients?

$$
v_t = \beta_2 v_{t-1} + (1 - \beta_2)\, g_t^2
$$

This is the same kind of average, of squared gradients, over roughly the last $\frac{1}{1 - 0.999} = 1000$ steps. $\sqrt{v_t}$ estimates the typical gradient size of that one parameter.

### The update

$$
\theta_t = \theta_{t-1} - \eta \, \frac{\hat m_t}{\sqrt{\hat v_t} + \epsilon}
$$

Dividing by $\sqrt{\hat v_t}$ rescales each parameter's step by its own typical gradient size. A parameter with tiny but steady gradients still moves about $\eta$ per step, and a parameter with huge gradients does not jump wildly. $\eta$ now sets the step size directly, whatever the scale of the gradient.

### Bias correction, derived

Because $m_0 = 0$, the early averages are pulled toward 0. Suppose the gradient is a constant $g$. Then

$$
\begin{aligned}
m_t &= (1 - \beta_1)\left(g + \beta_1 g + \beta_1^2 g + \dots + \beta_1^{t-1} g\right) \\
&= (1 - \beta_1)\, g \, \frac{1 - \beta_1^t}{1 - \beta_1} \\
&= (1 - \beta_1^t)\, g
\end{aligned}
$$

The middle line uses the geometric series $\sum_{s=0}^{t-1} \beta_1^s = \frac{1 - \beta_1^t}{1 - \beta_1}$. At $t = 1$, $m_1 = 0.1\,g$: ten times too small. Dividing by exactly that factor removes the bias:

$$
\hat m_t = \frac{m_t}{1 - \beta_1^t}, \qquad \hat v_t = \frac{v_t}{1 - \beta_2^t}
$$

The same argument gives $v_t = (1 - \beta_2^t) g^2$. As $t$ grows, $\beta^t \to 0$ and the correction fades out.

**Numbers at $t = 1$.** $\hat m_1 = \frac{0.1\,g}{0.1} = g$ and $\hat v_1 = \frac{0.001\,g^2}{0.001} = g^2$. So the first step is $\eta \frac{g}{|g|} = \pm\eta$, whatever the size of $g$. With $\eta = 0.01$ and gradients $10^{-4}$, $5$ and $-300$, all three entries move by exactly $0.01$ (the check tests this).

### What Adam remembers

For every parameter array, Adam keeps two arrays of the same shape, `m` and `v`, plus one shared step counter `t`. That triples the memory of the parameters, which is fine at our scale.
:::

::: think
Q: Without bias correction, how big is Adam's first step for a gradient $g$ (with $\beta_1 = 0.9$, $\beta_2 = 0.999$)?
A: $\frac{m_1}{\sqrt{v_1}} = \frac{0.1\,g}{\sqrt{0.001}\,|g|} = \frac{0.1}{0.0316} \approx 3.16$, in the direction of $g$. So the first steps are about 3 times larger than intended, which can upset a fresh model.
:::

::: think
Q: Two parameters have steady gradients of $10^{-4}$ and $1$. How far does each move per step with SGD at $\eta = 0.01$, and with Adam at $\eta = 0.01$?
A: SGD: $10^{-6}$ and $10^{-2}$, a factor of $10^4$ apart. Adam: about $0.01$ each, because $\frac{\hat m}{\sqrt{\hat v}} \approx \frac{g}{|g|} = 1$ for a steady gradient.
:::

::: think
Q: Why do $m$ and $v$ need the same shape as their parameter, while $t$ is a single number?
A: Every parameter entry has its own gradient history, so it needs its own averages. The step count is the same for all entries, because all are updated at every step, and it is only used in the correction factors.
:::

::: hint level=1
`__init__` stores the settings and creates arrays of zeros `m` and `v` for every parameter, plus `t = 0`. `step` adds 1 to `t`, updates both averages, corrects them, and moves each parameter in place.
:::

::: hint level=2
`np.zeros_like(p)` makes zeros with `p`'s shape and dtype. Use in-place operations (`*=`, `+=`, `-=`) for `m`, `v` and the parameter. `b1 ** t` is the bias factor. Read `self.lr` inside `step`, so that changing `opt.lr` between steps takes effect.
:::

::: hint level=3
```pseudo
__init__(params, lr, betas, eps):
    store params, lr, beta1, beta2, eps;  t ← 0
    m[name] ← zeros like p;  v[name] ← zeros like p   for each parameter

step(grads):
    t ← t + 1
    for each name, p:
        g ← grads[name]
        m[name] ← β1 · m[name] + (1 − β1) · g
        v[name] ← β2 · v[name] + (1 − β2) · g²
        m̂ ← m[name] / (1 − β1^t);   v̂ ← v[name] / (1 − β2^t)
        p −= lr · m̂ / (√v̂ + eps)                     # in place
```
:::

::: attempt
File: `workspace/stage1/optim.py` — fill in the `Adam` class.
Check: `python -m course check s1-l11`
:::

::: solution
@include solutions/stage1/optim.py#Adam

- `self.params = params` keeps the same dictionary and arrays the model uses; both SGD and Adam update them in place.
- `m *= b1; m += (1 - b1) * g` updates the stored arrays without making new ones each step.
- `1 - b1**t` is the bias-correction factor derived above. It is a Python number, so the arrays stay float32.
- `self.eps` sits outside the square root, as in the original paper. It only matters when $\hat v$ is almost 0.
:::

::: breakit
Remove the two bias-correction divisions and print how much each parameter changes in the first 3 steps of a fresh model. The first steps are about $3.16 \eta$ instead of $\eta$ (see the first think-question), and they shrink toward the corrected size over the first tens of steps.
:::

::: recap
- Momentum $m_t$ averages the gradients over about 10 steps; $v_t$ averages the squared gradients over about 1000.
- The update $\eta\,\hat m / (\sqrt{\hat v} + \epsilon)$ moves every parameter by about $\eta$.
- Bias correction divides by $1 - \beta^t$, derived from a geometric series with a constant gradient.
:::
