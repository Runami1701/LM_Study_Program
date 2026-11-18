---
id: s2-l08
title: AdamW, LR schedule, gradient clipping
stage: s2
order: 8
prereqs: [s2-l07]
files: [workspace/stage2/optim.py, workspace/stage2/schedule.py]
functions: [AdamW, clip_grad_norm, lr_at]
tests: tests/s2/test_l08_optim.py
est_minutes: 50
---

::: goal
You will write the three tools that make transformer training stable: AdamW (Adam with weight decay done correctly), a learning-rate schedule that rises at the start and then falls along a cosine curve, and gradient clipping, which shrinks a gradient that is too large.
:::

::: notation
| Symbol | Meaning |
| --- | --- |
| $\theta$ | a parameter entry |
| $g$ | its gradient |
| $\lambda$ | weight decay strength (`weight_decay`, 0.1 here) |
| $\eta$ | learning rate |
| $\hat m, \hat v$ | Adam's corrected averages (s1-l11) |
| $\epsilon$ | Adam's small constant |
| $t$ | step number |
| $t_w$ | number of warmup steps |
| $t_{\max}$ | the step where the schedule reaches its minimum |
| $\eta_{\max}, \eta_{\min}$ | highest and lowest learning rate |
| $\pi$ | 3.14159… |
| $c$ | the clipping threshold (`max_norm`) |
| $\lVert g \rVert$ | length of the whole gradient: $\sqrt{\text{sum of squares of all entries of all parameters}}$ |
:::

::: numpy
### Cosine from the math module

```python
import math
print(math.cos(0), math.cos(math.pi))      # 1.0 -1.0
```

### Length of a gradient spread over several arrays

Add the squares of every entry of every array, then take one square root:

```python
g1 = np.array([3., 0.])
g2 = np.array([[4.]])
print(np.sqrt((g1 ** 2).sum() + (g2 ** 2).sum()))   # 5.0
```

### Which parameters are matrices

`.ndim` is the number of axes: 2 for a weight matrix, 1 for a bias or a LayerNorm vector.

```python
print(np.zeros((3, 4)).ndim, np.zeros(4).ndim)   # 2 1
```

### Shrinking in place

`p.grad *= scale` multiplies the stored gradient in place, so the optimizer sees the shrunk version.
:::

::: concept
### Weight decay: two ways that look the same but aren't

Weight decay pulls every weight a little toward 0 at each step. This keeps weights small, which usually helps the model handle new text.

**Way 1: add it to the loss (L2).** Add $\frac{\lambda}{2}\theta^2$ to the loss. Its gradient is $\lambda\theta$, so Adam receives $g + \lambda\theta$ instead of $g$. But Adam divides everything by $\sqrt{\hat v}$, so the decay is divided too:

$$
\text{decay step} \approx \eta \, \frac{\lambda \theta}{\sqrt{\hat v} + \epsilon}
$$

Parameters with large gradients (large $\hat v$) get almost no decay; parameters with tiny gradients get a lot. That is not what we wanted.

**Way 2: decoupled (AdamW).** Keep the decay out of the gradient and apply it directly:

$$
\theta \leftarrow \theta - \eta \lambda \theta - \eta \, \frac{\hat m}{\sqrt{\hat v} + \epsilon}
$$

Now every weight shrinks by the same fraction $\eta\lambda$ per step, whatever its gradient. With $\eta = 10^{-3}$ and $\lambda = 0.1$, that is $10^{-4}$ of its value per step.

**Which parameters to decay.** We decay the matrices (`ndim >= 2`): the linear weights and the embedding tables. We do not decay biases and LayerNorm $\gamma$, $\beta$. They hold few numbers, and pulling $\gamma$ toward 0 would fight the normalization. (GPT-2 style training does the same.)

### The schedule: warmup, then cosine decay

At the very start, Adam's averages are based on a few steps and are unreliable, and the model can be damaged by large first steps. So the rate first rises in a straight line over $t_w$ warmup steps:

$$
\eta_t = \eta_{\max} \cdot \frac{t + 1}{t_w} \qquad (t < t_w)
$$

Then it falls along half a cosine wave to $\eta_{\min}$ at step $t_{\max}$:

$$
\eta_t = \eta_{\min} + \frac{1}{2}(\eta_{\max} - \eta_{\min})\left(1 + \cos\left(\pi \, \frac{t - t_w}{t_{\max} - t_w}\right)\right)
$$

Check the ends. At $t = t_w$ the fraction is 0, $\cos 0 = 1$, and $\eta = \eta_{\min} + (\eta_{\max} - \eta_{\min}) = \eta_{\max}$. At $t = t_{\max}$ the fraction is 1, $\cos \pi = -1$, and $\eta = \eta_{\min}$. Halfway, $\cos\frac{\pi}{2} = 0$ and $\eta = \frac{\eta_{\max} + \eta_{\min}}{2}$.

With the default config ($t_w = 200$, $t_{\max} = 5000$, $\eta_{\max} = 10^{-3}$, $\eta_{\min} = 10^{-4}$): step 0 uses $5 \times 10^{-6}$, step 199 uses $10^{-3}$, step 2600 uses $5.5 \times 10^{-4}$, and step 5000 uses $10^{-4}$. The cosine falls slowly at first, fast in the middle and slowly again at the end.

### Gradient clipping

Now and then one batch produces a huge gradient (an unusual text passage, or an unlucky step). One such step can undo many good ones. Clipping limits the length of the whole gradient to $c$:

$$
g \leftarrow g \cdot \min\left(1, \frac{c}{\lVert g \rVert}\right)
$$

$\lVert g \rVert$ is the length of **all** gradients together, as one long vector. If it is below $c$, nothing changes. Otherwise every gradient is multiplied by the same factor $\frac{c}{\lVert g \rVert}$, so the direction stays the same and only the length shrinks to $c$.

Example: gradients $[3, 0]$ and $[[4]]$ have length $\sqrt{9 + 0 + 16} = 5$. With $c = 1$ they become $[0.6, 0]$ and $[[0.8]]$, which have length 1. `clip_grad_norm` returns the length **before** clipping; logging it shows you when training gets rough.
:::

::: think
Q: With L2 weight decay inside Adam, which parameters get decayed least, and why is that backwards?
A: Those with the largest gradients, because Adam divides the decay by $\sqrt{\hat v}$, which is large for them. But weight decay is meant to act the same on every weight, independent of how big its gradient happens to be. AdamW applies it directly, so every weight shrinks by the same fraction.
:::

::: think
Q: Why does clipping scale all gradients by the same factor instead of cutting each entry at $\pm c$?
A: The same factor keeps the direction of the whole gradient, so the step still points downhill; only its length shrinks. Cutting entries separately changes the direction: large entries are cut and small ones are not.
:::

::: think
Q: What learning rate does the default schedule use at step 100, and why so small?
A: $10^{-3} \cdot \frac{101}{200} \approx 5 \times 10^{-4}$, half the maximum. It is still warming up: Adam's averages are based on few steps and the model is fragile, so the steps grow gradually.
:::

::: hint level=1
`AdamW` is your s1-l11 Adam with two changes: it reads gradients from `p.grad` (Tensors), and it shrinks matrices by `lr * weight_decay` before the Adam step. `lr_at` has three cases. `clip_grad_norm` computes one length, then scales.
:::

::: hint level=2
In `AdamW.step`: `if p.data.ndim >= 2: p.data -= self.lr * self.weight_decay * p.data`. In `lr_at`: `if step < warmup: ...`, `if step >= max_steps: return lr_min`, else the cosine with `math.cos`. In `clip_grad_norm`: add `(p.grad ** 2).sum()` over all parameters, take `math.sqrt`, and multiply each `p.grad` in place when the length exceeds `max_norm`.
:::

::: hint level=3
```pseudo
AdamW.step():
    t ← t + 1
    for each name, p:
        update m and v from p.grad, as in Adam
        if p is a matrix (ndim ≥ 2): p.data −= lr · weight_decay · p.data
        p.data −= lr · m̂ / (√v̂ + eps)

lr_at(step, warmup, max_steps, lr_max, lr_min):
    if step < warmup: return lr_max · (step + 1) / warmup
    if step ≥ max_steps: return lr_min
    progress ← (step − warmup) / (max_steps − warmup)
    return lr_min + 0.5 · (lr_max − lr_min) · (1 + cos(π · progress))

clip_grad_norm(params, max_norm):
    total ← square root of the sum of all squared gradient entries
    if total > max_norm: multiply every p.grad by max_norm / (total + 1e-6)
    return total
```
:::

::: attempt
Files: `workspace/stage2/optim.py` — fill in `AdamW` and `clip_grad_norm`; `workspace/stage2/schedule.py` — fill in `lr_at`.
Check: `python -m course check s2-l08`
:::

::: solution
@include solutions/stage2/optim.py#AdamW

- The Adam part is unchanged from s1-l11, but reads `p.grad` from the Tensors.
- The decay line runs before the Adam step and only for matrices; it is not part of `m` or `v`.

@include solutions/stage2/optim.py#clip_grad_norm

- The sum is computed in float64, so the squares of many small numbers don't lose precision.
- `scale < 1.0` only when the length exceeds `max_norm`; then every gradient shrinks by the same factor.

@include solutions/stage2/schedule.py#lr_at
:::

::: breakit
Train the full model (s2-l09) with `warmup=1`, `lr_max=3e-3` and `grad_clip=1e9` (no clipping). Watch the first 50 losses. They jump up and down or even rise, and the run may never recover the loss that the default settings reach. Then restore `warmup=200` and `grad_clip=1.0`.
:::

::: recap
- AdamW shrinks matrices by $\eta\lambda\theta$ directly; L2 inside Adam would decay large-gradient weights least.
- Warmup raises the rate in a straight line, then a half cosine lowers it from $\eta_{\max}$ to $\eta_{\min}$.
- Clipping scales the whole gradient to length at most $c$ and keeps its direction.
:::
