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
Implement the Adam optimizer, with momentum, per-parameter step sizes and bias correction, as a class whose `step(grads)` updates every parameter in place.
:::

::: notation
| Symbol | Meaning |
| --- | --- |
| $g_t$ | the gradient at step $t$ (for one parameter entry) |
| $t$ | the step count, starting at $t = 1$ |
| $m_t$ | first moment: exponential moving average of gradients ("momentum") |
| $v_t$ | second moment: exponential moving average of squared gradients |
| $\beta_1, \beta_2$ | decay rates of $m$ and $v$; defaults $0.9$ and $0.999$ |
| $\hat m_t, \hat v_t$ | bias-corrected moments |
| $\eta$ | learning rate |
| $\epsilon$ | a tiny constant ($10^{-8}$) that avoids division by zero |
| $\theta$ | a parameter entry |
| $g$ | a constant gradient (in the bias-correction argument) |
:::

::: concept
Plain SGD has two weaknesses. Its direction is noisy (every minibatch disagrees a bit), and one learning rate must serve every parameter, although some get gradients of size $10^{-4}$ and others of size $1$. Adam fixes both.

### Momentum: an exponential moving average

$$
m_t = \beta_1 m_{t-1} + (1 - \beta_1)\, g_t, \qquad m_0 = 0
$$

Expand the first three steps with $\beta_1 = 0.9$:

$$
\begin{aligned}
m_1 &= 0.1\, g_1 \\
m_2 &= 0.9 \cdot 0.1\, g_1 + 0.1\, g_2 = 0.09\, g_1 + 0.1\, g_2 \\
m_3 &= 0.081\, g_1 + 0.09\, g_2 + 0.1\, g_3
\end{aligned}
$$

Each older gradient is weighted 0.9 times less: a smooth average over roughly the last $\frac{1}{1 - \beta_1} = 10$ steps. Noise that flips sign between batches averages out; a consistent direction builds up.

### Second moment: how big are this parameter's gradients?

$$
v_t = \beta_2 v_{t-1} + (1 - \beta_2)\, g_t^2
$$

the same average, of squared gradients, over roughly the last $\frac{1}{1 - 0.999} = 1000$ steps. $\sqrt{v_t}$ is a running estimate of the typical gradient size of that one parameter.

### The update

$$
\theta_t = \theta_{t-1} - \eta \, \frac{\hat m_t}{\sqrt{\hat v_t} + \epsilon}
$$

Dividing by $\sqrt{\hat v_t}$ rescales each parameter's step by its own typical gradient size: a parameter with tiny but consistent gradients still moves about $\eta$ per step, and one with huge gradients doesn't jump wildly. The step size is set by $\eta$ in parameter units, not by the gradient's scale.

### Bias correction, derived

Because $m_0 = 0$, the early averages are pulled toward 0. Suppose the gradient is a constant $g$. Then

$$
\begin{aligned}
m_t &= (1 - \beta_1)\left(g + \beta_1 g + \beta_1^2 g + \dots + \beta_1^{t-1} g\right) \\
&= (1 - \beta_1)\, g \, \frac{1 - \beta_1^t}{1 - \beta_1} \\
&= (1 - \beta_1^t)\, g
\end{aligned}
$$

(the middle line is the geometric series $\sum_{s=0}^{t-1} \beta_1^s = \frac{1 - \beta_1^t}{1 - \beta_1}$). At $t = 1$, $m_1 = 0.1\,g$: ten times too small. Dividing by exactly that factor removes the bias:

$$
\hat m_t = \frac{m_t}{1 - \beta_1^t}, \qquad \hat v_t = \frac{v_t}{1 - \beta_2^t}
$$

The same argument gives $v_t = (1 - \beta_2^t) g^2$ for $v$. As $t$ grows, $\beta^t \to 0$ and the correction fades away.

**Numbers at $t = 1$.** $\hat m_1 = \frac{0.1\,g}{0.1} = g$ and $\hat v_1 = \frac{0.001\,g^2}{0.001} = g^2$, so the first step is $\eta \frac{g}{|g|} = \pm\eta$, whatever the size of $g$. With $\eta = 0.01$ and gradients $10^{-4}$, $5$ and $-300$, all three entries move by exactly $0.01$ (the check tests this).

### State

Adam keeps, for every parameter array, two arrays of the same shape (`m` and `v`) and one shared step counter `t`. That triples the memory of the parameters, which is fine at our scale.
:::

::: think
Q: Without bias correction, how big is Adam's first step for a gradient $g$ (with $\beta_1 = 0.9$, $\beta_2 = 0.999$)?
A: $\frac{m_1}{\sqrt{v_1}} = \frac{0.1\,g}{\sqrt{0.001}\,|g|} = \frac{0.1}{0.0316} \approx 3.16$ times $\eta$ in the sign of $g$. The early steps are about 3× larger than intended, which can destabilize a fresh model.
:::

::: think
Q: Two parameters have gradients that are consistently $10^{-4}$ and $10^{0}$. How far does each move per step under SGD with $\eta = 0.01$, and under Adam with $\eta = 0.01$?
A: SGD: $10^{-6}$ and $10^{-2}$, a factor of $10^4$ apart. Adam: about $0.01$ each, since $\frac{\hat m}{\sqrt{\hat v}} \approx \frac{g}{|g|} = 1$ for a consistent gradient.
:::

::: think
Q: Why do $m$ and $v$ need the same shape as their parameter, while $t$ is a single number?
A: Every parameter entry has its own gradient history, so its own averages. The step count is the same for all entries (they are all updated every step), and it is only used in the bias-correction factors.
:::

::: hint level=1
`__init__` stores the settings and creates zero-filled `m` and `v` arrays for every parameter, plus `t = 0`. `step` increments `t`, updates both averages, corrects them, and moves each parameter in place.
:::

::: hint level=2
`np.zeros_like(p)` makes a zero array with `p`'s shape and dtype. Use in-place operations (`*=`, `+=`, `-=`) for `m`, `v` and the parameter. `b1**t` is the bias factor. Read `self.lr` inside `step`, so changing `opt.lr` between steps takes effect.
:::

::: hint level=3
```pseudo
__init__(params, lr, betas, eps):
    remember params, lr, beta1, beta2, eps;  t ← 0
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
File: `workspace/stage1/optim.py` — implement the `Adam` class.
Check: `python -m course check s1-l11`
:::

::: solution
@include solutions/stage1/optim.py#Adam

- `self.params = params` keeps the same dict and arrays the model uses, which is why `sgd_step` and Adam both update in place.
- `m *= b1; m += (1 - b1) * g` updates the state arrays without allocating new ones each step.
- `1 - b1**t` is the bias-correction factor derived above; it is a Python float, so the arrays stay float32.
- `self.eps` sits outside the square root, as in the original paper; it only matters when $\hat v$ is nearly 0.
:::

::: breakit
Remove the two bias-correction divisions and print each parameter's change on the first 3 steps of a fresh model. Expect first steps about $3.16 \eta$ in size instead of $\eta$ (see the first think-question), shrinking toward the corrected size over the first tens of steps.
:::

::: recap
- Momentum $m_t$ averages gradients (≈ last 10 steps); $v_t$ averages squared gradients (≈ last 1000).
- The update $\eta\,\hat m / (\sqrt{\hat v} + \epsilon)$ gives every parameter a step of about $\eta$.
- Bias correction divides by $1 - \beta^t$, derived from a constant-gradient geometric series.
:::
