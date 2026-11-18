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
You will write the training loop. Each step draws a batch, runs the forward and backward pass, moves every parameter a little against its gradient, $\theta \leftarrow \theta - \eta \nabla_\theta L$, and records the loss.
:::

::: notation
| Symbol | Meaning |
| --- | --- |
| $\theta$ | the parameters (one entry, or all of them) |
| $\eta$ | learning rate: how big a step to take, e.g. $0.1$ |
| $\nabla_\theta L$ | the gradient: all the $\frac{\partial L}{\partial \theta}$ for every parameter entry |
| $L$ | the loss on the current batch |
| $t$ | the step number |
| $B$ | batch size |
| $N$ | number of training positions (about 1.0M for Tiny Shakespeare) |
| $L_{\text{full}}$ | the loss averaged over all $N$ training examples |
| $\mathrm{Var}$ | variance (s1-l09) |
| $\lVert g \rVert$ | the length of a vector: $\lVert g \rVert^2 = \sum_i g_i^2$ |
:::

::: numpy
### Change an array in place, or replace it

`W -= 0.5` changes the existing array. `W = W - 0.5` builds a new array and points the name `W` at it; anything that still refers to the old array does not see the change:

```python
W = np.array([1., 2.])
ref = W                      # a second name for the same array
W -= 0.5                     # in place
print(ref, ref is W)         # [0.5 1.5] True

W2 = np.array([1., 2.])
ref2 = W2
W2 = W2 - 0.5                # a new array
print(ref2, ref2 is W2)      # [1. 2.] False   ref2 never saw the update
```

`a is b` asks whether two names refer to the same object.

### A loop over a dictionary

`for name, p in params.items():` visits each parameter name and array. Inside the loop, `p -= lr * grads[name]` updates the array stored in the dictionary.
:::

::: concept
### Gradient descent

The gradient points in the direction in which the loss grows fastest. So step the other way:

$$
\theta_{t+1} = \theta_t - \eta \, \nabla_\theta L(\theta_t)
$$

Why this lowers the loss: for a small step $\Delta\theta$, $L(\theta + \Delta\theta) \approx L(\theta) + \nabla_\theta L \cdot \Delta\theta$. With $\Delta\theta = -\eta \nabla_\theta L$:

$$
\begin{aligned}
L(\theta_{t+1}) &\approx L(\theta_t) - \eta\, \nabla_\theta L \cdot \nabla_\theta L \\
&= L(\theta_t) - \eta \, \lVert \nabla_\theta L \rVert^2 \le L(\theta_t)
\end{aligned}
$$

because $\lVert g \rVert^2 \ge 0$. This only holds while $\eta$ is small enough for the approximation to stay accurate. Too large a step jumps over the valley (see Break it).

### Why batches

The loss we really care about is the average over all $N \approx 10^6$ training positions, $L_{\text{full}}$. Its exact gradient needs a forward and backward pass over the whole text, for **one** step. A batch of $B = 128$ random examples gives a noisy estimate of the same gradient at $\frac{128}{10^6}$ of the cost. On average the estimate is right, and its noise shrinks like $\frac{1}{\sqrt{B}}$: the variance of a mean of $B$ independent samples is $\frac{\mathrm{Var}}{B}$. Many cheap, slightly noisy steps beat a few exact ones.

### Steps and epochs

An **epoch** is one pass over the training data. With random batches there is no exact pass, but the equivalent count is

$$
\text{steps per epoch} = \frac{N}{B} = \frac{1{,}003{,}854}{128} \approx 7{,}843
$$

so 40,000 steps are about 5 epochs. We count **steps**, because the learning-rate schedule and the logs use them.

### Update in place

`sgd_step` must change each array in place (`p -= lr * g`), not build new arrays (`params[name] = p - lr * g`). Other objects keep references to the parameter arrays: the Adam optimizer of s1-l11 keeps the dictionary, and a cache may hold `W`. In-place updates keep them all looking at the same memory (see the box above).

### The loop

Each step: draw a batch, run the forward pass (loss and caches), run the backward pass (gradients), update, record the loss. That is all. Recording every step's loss lets you plot a curve later. Single-batch losses jump around, so plots usually show a moving average.

`train(config, train_data, val_data, V)` returns a dictionary with the final parameters and the loss histories. The check runs it with a tiny model on a perfectly predictable sequence (`0, 1, ..., 19, 0, 1, ...`), where the loss must fall from $\ln 20 \approx 3.0$ to near 0 in 200 steps. The evaluation (`eval_every`) and the learning-rate schedule (`lr_at`) are already part of the recipe; you fill them in during s1-l12. Until then, `lr_at` returns the constant `config["lr"]`.
:::

::: think
Q: Why not compute the exact gradient over all 1M training positions at every step?
A: It costs about 8,000 times more per step than a batch of 128, and one exact step is not worth 8,000 noisy ones: the noisy gradient already points roughly downhill. Batches trade a little accuracy per step for many more steps.
:::

::: think
Q: Your `sgd_step` does `params[name] = params[name] - lr * grads[name]`. The loss still goes down. What breaks later?
A: The dictionary gets new arrays every step. Anything that kept the old arrays, such as an optimizer's memory or a name taken before training (`W1 = params["W1"]`), keeps seeing stale values. It also allocates a fresh array per parameter at every step.
:::

::: think
Q: What should the loss at step 0 be in the check's tiny model (20 characters), and why can its loss reach almost 0 when a Shakespeare model's cannot?
A: About $\ln 20 \approx 3.0$. The sequence `0..19` is fully determined by the previous number, so the model can become certain ($p_y \to 1$, loss $\to 0$). Shakespeare's next character is genuinely uncertain, so even a perfect model keeps a loss well above 0.
:::

::: hint level=1
`sgd_step` is one line inside a loop over the parameter names. `train` follows the numbered recipe in its docstring: one `rng`, one call to `init_params`, then a short step inside a `for` loop.
:::

::: hint level=2
Subtract in place with `-=`. In `train`, make the generator with `np.random.default_rng(config["seed"])` and use it for both `init_params` and every `get_batch`. Append each step's loss to a Python list. Create `Adam` only when `config["optimizer"] == "adam"`.
:::

::: hint level=3
```pseudo
sgd_step(params, grads, lr):
    for each name, p in params:  p −= lr · grads[name]      # in place

train(config, train_data, val_data, V):
    rng ← default_rng(config seed);  params ← init_params(V, k, e, h, rng)
    opt ← Adam(params, lr) if the config optimizer is "adam", otherwise none
    train_loss, val_loss ← [], []
    for step in 0 .. steps−1:
        lr ← lr_at(step, config)
        x, y ← get_batch(train_data, B, k, rng)
        loss, cache ← forward(params, x, y);  grads ← backward(cache)
        if opt: opt.lr ← lr; opt.step(grads)    otherwise: sgd_step(params, grads, lr)
        append loss to train_loss
        if eval_every and (step is a multiple of eval_every or the last step):
            v ← estimate_loss(params, val_data, B, k, eval_batches, rng); append (step, v); log it
    return {params, train_loss, val_loss}
```
:::

::: attempt
The editor below has a tab for each file. In `workspace/stage1/optim.py`: fill in `sgd_step`; in `workspace/stage1/train_mlp.py`, fill in `train`.
Check: press **Run check** in the Check section below.
:::

::: solution
@include solutions/stage1/optim.py#sgd_step

- `p -= lr * grads[name]` writes into the existing array, so every reference to it sees the update. `lr` is a Python number, so float32 stays float32.

@include solutions/stage1/train_mlp.py#train

- One `rng`, seeded from the config, drives both the starting values and the batch draws, so a run can be repeated exactly.
- The optimizer branch keeps the loop the same for SGD and Adam. Setting `opt.lr` at each step lets the schedule of s1-l12 control Adam too.
- Evaluation runs on the first step, every `eval_every` steps and on the last step, so the log shows where training started and where it ended.
:::

::: breakit
Train the Shakespeare model with SGD (`optimizer="sgd"`, `lr_final=None`, 3,000 steps, `eval_every=0`) at $\eta = 10$, $\eta = 0.1$ and $\eta = 0.001$, and plot the three loss curves (moving average over 50 steps). On the reference run, $\eta = 10$ shot up into the hundreds and stayed above 50, far worse than guessing; $\eta = 0.001$ crawled from 4.14 to 3.21; and $\eta = 0.1$ fell steadily to about 2.13.
:::

::: recap
- SGD: $\theta \leftarrow \theta - \eta \nabla_\theta L$; to first order it lowers the loss by $\eta \lVert\nabla L\rVert^2$.
- Batches give cheap, noisy gradients that are right on average; the noise falls like $1/\sqrt{B}$.
- One epoch of Tiny Shakespeare is about 7,843 steps at $B = 128$.
- Updates happen in place, and one seeded generator makes training repeatable.
:::
