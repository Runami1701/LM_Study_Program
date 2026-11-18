---
id: s1-l12
title: Evaluation, LR sweep, hitting the target
stage: s1
order: 12
prereqs: [s1-l11]
files: [workspace/stage1/train_mlp.py]
functions: [estimate_loss, lr_at]
tests: tests/s1/test_l12_eval.py
est_minutes: 60
---

::: goal
Measure the model honestly: average the loss over many validation batches, read the train/validation gap, choose a learning rate with a range test, decay it over the run, and produce `runs/s1/results.json` for the boss check.
:::

::: notation
| Symbol | Meaning |
| --- | --- |
| $L_i$ | the loss of the $i$-th sampled batch |
| $n$ | number of batches averaged (`n_batches`) |
| $\hat L$ | the estimate $\frac{1}{n}\sum_i L_i$ |
| $\sigma$ | standard deviation of a single batch's loss |
| $\mathrm{Var}$ | variance |
| $\eta_0, \eta_1$ | learning rate at the first and at the last step (`lr`, `lr_final`) |
| $\eta_t$ | learning rate at step $t$ |
| $t$ | step index, $0 \dots S-1$ |
| $S$ | total number of steps (`steps`) |
:::

::: concept
### One batch is not a measurement

The loss on a single batch of 128 examples is noisy. For the trained reference model, 500 validation batches gave losses from 1.21 to 2.12, with standard deviation $\sigma \approx 0.15$. Averaging $n$ independent batches divides the variance by $n$:

$$
\mathrm{Var}(\hat L) = \mathrm{Var}\left(\frac{1}{n} \sum_{i=1}^{n} L_i\right) = \frac{1}{n^2} \cdot n\,\sigma^2 = \frac{\sigma^2}{n}
$$

so the standard deviation of the estimate is $\frac{\sigma}{\sqrt{n}}$. With $n = 50$: $\frac{0.15}{\sqrt{50}} \approx 0.021$ (measured: 0.023). That is precise enough to compare runs differing by 0.05; one batch is not. `estimate_loss` draws $n$ batches with `get_batch` and returns their mean. The final number in `results.json` uses 200 batches of 512 examples.

### The train/validation gap

Training loss measures what the model has memorized; validation loss measures what it generalizes. The reference run ends with **train 1.37, validation 1.68**. A gap that keeps widening while validation stalls or rises is **overfitting**: 134K parameters can memorize parts of a 1M-character text. Remedies: stop earlier, use a smaller model, or get more data (stage 3). A model that underfits shows train ≈ validation, both still falling: train longer or make it bigger.

### Choosing a learning rate: the range test

Run short trainings (1,500 Adam steps here) at log-spaced learning rates and compare the final losses. The reference results, mean training loss over the last 100 steps:

| learning rate | $10^{-4}$ | $3 \cdot 10^{-4}$ | $10^{-3}$ | $3 \cdot 10^{-3}$ | $10^{-2}$ | $3 \cdot 10^{-2}$ | $10^{-1}$ |
| --- | --- | --- | --- | --- | --- | --- | --- |
| loss | 2.50 | 2.24 | 2.00 | **1.98** | 2.44 | 4.61 | 15.4 |

Too small crawls; too large becomes unstable and then diverges. The best short-run value ($3 \cdot 10^{-3}$) is a good **starting** rate. Log spacing matters: learning rates act multiplicatively, so $10^{-3}$ vs $3 \cdot 10^{-3}$ is as big a step as $10^{-2}$ vs $3 \cdot 10^{-2}$.

### Decay: big steps early, small steps late

Early on, large steps cover ground fast. Late in training, the noise of minibatch gradients keeps a large-step model bouncing around a minimum instead of settling into it. So we shrink the rate over the run, exponentially from $\eta_0$ to $\eta_1$:

$$
\eta_t = \eta_0 \left(\frac{\eta_1}{\eta_0}\right)^{t/(S-1)}
$$

Check the endpoints: $t = 0$ gives exponent 0 and $\eta_0$; $t = S-1$ gives exponent 1 and $\eta_0 \cdot \frac{\eta_1}{\eta_0} = \eta_1$. Halfway, the exponent is $\frac{1}{2}$:

$$
\eta_{\text{mid}} = \eta_0 \left(\frac{\eta_1}{\eta_0}\right)^{1/2} = \sqrt{\eta_0\, \eta_1}
$$

the geometric mean: $10^{-2}$ and $10^{-4}$ meet at $10^{-3}$. Each step multiplies the rate by the same factor, which suits a quantity that acts multiplicatively.

How much it matters here: the default run (Adam, $3 \cdot 10^{-3}$ decaying to $3 \cdot 10^{-4}$) ends at validation **1.68**; the same run at a constant $3 \cdot 10^{-3}$ ends at **1.90**.

### results.json

`main()` in `train_mlp.py` (given) trains with `CONFIG`, then writes `runs/s1/results.json`: final validation and training loss (200 batches of 512), wall time, step count, parameter count, the config and a sample. It also saves the parameters and vocabulary to `runs/s1/params.npz`. Logging every run's config next to its result is how you keep track of what you tried.

The reference: validation **1.682** in **1.4 minutes** on a 4-core cloud CPU. The boss target is 1.75.
:::

::: think
Q: Your evaluation uses 5 batches and reports 1.70 for run A and 1.74 for run B. Is A better?
A: Not established. With $\sigma \approx 0.15$, five batches give a standard error of $\frac{0.15}{\sqrt 5} \approx 0.07$, larger than the difference. Re-evaluate with 50+ batches (error ~0.02) before concluding anything.
:::

::: think
Q: Validation loss goes 2.03, 1.89, 1.80, 1.72, 1.70, 1.71, 1.70 while training loss keeps falling to 1.37. What is happening, and what would you change first?
A: Overfitting: the model keeps improving on the text it has seen but no longer on new text. Training longer won't help. Options: fewer steps (stop around the minimum), a smaller hidden layer, or more data. For the boss target, the current curve is already fine.
:::

::: think
Q: In the range test, why judge by the loss after a fixed number of steps rather than after "convergence"?
A: Convergence takes the full run for every candidate, which is expensive. The early-run comparison captures what the learning rate mostly controls: whether training is stable and how fast it makes progress. The decay schedule then handles the late phase.
:::

::: hint level=1
`estimate_loss` is a short loop: draw a batch, compute its loss, collect, average. `lr_at` is one formula with an exponent that goes from 0 to 1 over the run.
:::

::: hint level=2
Call `get_batch(data, B, k, rng)` exactly `n_batches` times, `forward(params, x, y)[0]` for each loss, and return `float(np.mean(...))`. For `lr_at`, the fraction is `step / (config["steps"] - 1)`; return `config["lr"]` unchanged when `lr_final` is `None`.
:::

::: hint level=3
```pseudo
estimate_loss(params, data, B, k, n_batches, rng):
    losses ← []
    repeat n_batches times:
        x, y ← get_batch(data, B, k, rng);  append forward(params, x, y) loss
    return mean(losses) as a Python float

lr_at(step, config):
    if lr_final is None: return lr
    frac ← step / (steps − 1)
    return lr · (lr_final / lr) ^ frac
```
:::

::: attempt
File: `workspace/stage1/train_mlp.py` — implement `estimate_loss` and `lr_at`.
Check: `python -m course check s1-l12`
Then run the full training (about 1.5 minutes on the reference CPU): `python -m workspace.stage1.train_mlp`, and take on the boss: `python -m course check s1-boss`.
:::

::: solution
@include solutions/stage1/train_mlp.py#estimate_loss

- Drawing through `get_batch` with the caller's `rng` makes the estimate reproducible for a fixed seed.
- Returning `float(np.mean(...))` gives a plain number for logs and JSON.

@include solutions/stage1/train_mlp.py#lr_at

- `max(1, steps - 1)` avoids dividing by zero for a one-step run; `not lr1` treats `None` (and 0) as "no decay".
:::

::: breakit
Evaluate the same trained model with `n_batches = 1` ten times (different seeds) and with `n_batches = 50` ten times. Expect the single-batch values to spread over roughly ±0.3 (e.g. 1.45 to 1.95) and the 50-batch values to agree within about ±0.05.
:::

::: breakit
Go back to the break-it of s1-l01: build the split by shuffling the windows (so val windows overlap training ones), train, and compare. Expect the "validation" loss to come out clearly below the honest 1.68 and close to the training loss: the leak makes the model look like it generalizes better than it does.
:::

::: recap
- Average many batches: the estimate's noise falls like $\sigma / \sqrt{n}$.
- The train/validation gap diagnoses overfitting vs underfitting.
- A log-spaced range test picks the starting learning rate; exponential decay to `lr_final` gains 0.2 nats here.
- `results.json` records every run's numbers and config.
:::
