---
id: s1-l10
title: Training loop + SGD
stage: s1
order: 10
prereqs: [s1-l09]
files: [workspace/stage1/optim.py, workspace/stage1/train_mlp.py]
functions: [sgd_step, train]
tests: tests/s1/test_l10_train.py
est_minutes: 45
---

::: goal
Write the minibatch training loop: sample a batch, forward, backward, update every parameter with SGD, $\theta \leftarrow \theta - \eta \nabla_\theta L$, and record the loss at every step.
:::

::: notation
| Symbol | Meaning |
| --- | --- |
| $\theta$ | all parameters (any one entry, or all of them together) |
| $\eta$ | learning rate (step size), e.g. $0.1$ |
| $\nabla_\theta L$ | the gradient: the array of $\frac{\partial L}{\partial \theta}$ for every parameter entry |
| $L$ | the loss on the current minibatch |
| $t$ | the step number |
| $B$ | batch size |
| $N$ | number of training positions ($\approx 1.0$M for Tiny Shakespeare) |
| $L_{\text{full}}$ | the loss averaged over all $N$ training examples |
| $\mathrm{Var}$ | variance (s1-l09) |
:::

::: concept
### Gradient descent

The gradient points in the direction in which the loss increases fastest. Step the other way:

$$
\theta_{t+1} = \theta_t - \eta \, \nabla_\theta L(\theta_t)
$$

Why it lowers the loss, to first order: for a small step $\Delta\theta$, $L(\theta + \Delta\theta) \approx L(\theta) + \nabla_\theta L \cdot \Delta\theta$. With $\Delta\theta = -\eta \nabla_\theta L$:

$$
\begin{aligned}
L(\theta_{t+1}) &\approx L(\theta_t) - \eta\, \nabla_\theta L \cdot \nabla_\theta L \\
&= L(\theta_t) - \eta \, \lVert \nabla_\theta L \rVert^2 \le L(\theta_t)
\end{aligned}
$$

where $\lVert g \rVert^2 = \sum g_i^2 \ge 0$ is the squared length of the gradient. This holds only while $\eta$ is small enough for the linear approximation to stay accurate; too large a step overshoots the valley (see Break it).

### Why minibatches

The loss we really care about averages over all $N \approx 10^6$ training positions, $L_{\text{full}}$. Its exact gradient costs a forward and backward pass over the whole text, for **one** step. A minibatch of $B = 128$ random examples gives a noisy but unbiased estimate of the same gradient for $\frac{128}{10^6}$ of the cost. The noise shrinks like $\frac{1}{\sqrt{B}}$ (the variance of a mean of $B$ independent samples is $\frac{\mathrm{Var}}{B}$). Many cheap, slightly noisy steps beat a few exact ones, and the noise even helps escape poor regions.

### Steps and epochs

An **epoch** is one pass over the training data. With random sampling there is no exact pass, but the equivalent count is

$$
\text{steps per epoch} = \frac{N}{B} = \frac{1{,}003{,}854}{128} \approx 7{,}843
$$

so 40,000 steps is about 5 epochs. We count **steps**, because that is what the learning-rate schedule and the logs use.

### In place, and why it matters

`sgd_step` should modify each array in place (`p -= lr * g`), not build new arrays (`params[name] = p - lr * g`). Other objects hold references to the parameter arrays: the Adam optimizer of s1-l11 keeps the dict, a cache may hold `W`. In-place updates keep everyone looking at the same memory.

### The loop

Per step: get a batch, forward (loss and caches), backward (gradients), update, record the loss. That is all. Logging the raw per-step losses lets you plot a curve later; individual minibatch losses jump around, so plots usually show a moving average.

`train(config, train_data, val_data, V)` returns a dict with the final parameters and the histories. The checks call it with a tiny config on a perfectly predictable sequence (`0, 1, ..., 19, 0, 1, ...`), where the loss must fall from $\ln 20 \approx 3.0$ to near 0 in 200 steps. Evaluation (`eval_every`) and the learning-rate schedule (`lr_at`) are wired in now but filled in by s1-l12; until then `lr_at` returns the constant `config["lr"]`.
:::

::: think
Q: Why not compute the exact gradient over all 1M training positions at each step?
A: It costs ~8,000 times more per step than a batch of 128, and one exact step is not worth 8,000 noisy ones: the noisy gradient already points roughly downhill. Minibatches trade a little accuracy per step for vastly more steps.
:::

::: think
Q: Your `sgd_step` does `params[name] = params[name] - lr * grads[name]`. The loss still goes down in your loop. What breaks later?
A: The dict now holds new arrays each step. Anything holding the old arrays (an optimizer's state keyed to them, a cache, a reference taken before training such as `W1 = params["W1"]`) keeps seeing stale values. It also allocates a fresh array per parameter every step.
:::

::: think
Q: What should the loss at step 0 be in the check's tiny config (20 classes), and why is a near-zero final loss possible there but not on Shakespeare?
A: About $\ln 20 \approx 3.0$. The sequence `0..19` repeating is fully determined by the previous character, so the model can become certain ($p_y \to 1$, loss $\to 0$). Shakespeare's next character is genuinely uncertain, so even a perfect model has a loss well above 0.
:::

::: hint level=1
`sgd_step` is one line inside a loop over the parameter names. `train` follows the numbered recipe in its docstring: one `rng`, one init, then the five-line step inside a `for` loop.
:::

::: hint level=2
In-place subtraction is `-=`. In `train`, make the rng with `np.random.default_rng(config["seed"])` and use it for both `init_params` and every `get_batch`. Append each step's loss to a Python list. Create `Adam` only when `config["optimizer"] == "adam"`.
:::

::: hint level=3
```pseudo
sgd_step(params, grads, lr):
    for each name, p in params:  p −= lr · grads[name]      # in place

train(config, train_data, val_data, V):
    rng ← default_rng(config seed);  params ← init_params(V, k, e, h, rng)
    opt ← Adam(params, lr) if config optimizer is "adam", else none
    train_loss, val_loss ← [], []
    for step in 0 .. steps−1:
        lr ← lr_at(step, config)
        x, y ← get_batch(train_data, B, k, rng)
        loss, cache ← forward(params, x, y);  grads ← backward(cache)
        if opt: opt.lr ← lr; opt.step(grads)   else: sgd_step(params, grads, lr)
        append loss to train_loss
        if eval_every and (step divisible by eval_every or last step):
            v ← estimate_loss(params, val_data, B, k, eval_batches, rng); append (step, v); log
    return {params, train_loss, val_loss}
```
:::

::: attempt
Files: `workspace/stage1/optim.py` — implement `sgd_step`; `workspace/stage1/train_mlp.py` — implement `train`.
Check: `python -m course check s1-l10`
:::

::: solution
@include solutions/stage1/optim.py#sgd_step

- `p -= lr * grads[name]` writes into the existing array, so every reference to it sees the update; `lr` is a Python float, so float32 stays float32.

@include solutions/stage1/train_mlp.py#train

- One `rng`, seeded from the config, drives both initialization and batch sampling, so a run is exactly reproducible.
- The optimizer branch keeps the loop identical for SGD and Adam; setting `opt.lr` each step lets the schedule of s1-l12 drive Adam too.
- Evaluation runs on the first step, every `eval_every` steps, and on the last step, so the log always shows where training started and ended.
:::

::: breakit
Train the full Shakespeare model with SGD (`optimizer="sgd"`, `lr_final=None`, 3,000 steps, `eval_every=0`) at $\eta = 10$, $\eta = 0.1$ and $\eta = 0.001$, and plot the three loss curves (moving average over 50 steps). Expect (reference run): $\eta = 10$ shoots up into the hundreds and stays above 50, far worse than guessing; $\eta = 0.001$ crawls from 4.14 to 3.21; $\eta = 0.1$ falls steadily to about 2.13.
:::

::: recap
- SGD: $\theta \leftarrow \theta - \eta \nabla_\theta L$; to first order it lowers the loss by $\eta \lVert\nabla L\rVert^2$.
- Minibatches give cheap, unbiased, noisy gradients; noise falls like $1/\sqrt{B}$.
- About 7,843 steps make one epoch of Tiny Shakespeare at $B = 128$.
- Updates are in place, and one seeded rng makes training reproducible.
:::
