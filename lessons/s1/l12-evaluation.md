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
jobs: [train-s1]
---

::: goal
You will measure the model honestly. You average the loss over many validation batches, read the gap between training and validation loss, pick a learning rate with a range test, and lower the learning rate during the run. The final run writes `runs/s1/results.json` for the boss check.
:::

::: notation
| Symbol | Meaning |
| --- | --- |
| $L_i$ | the loss of the $i$-th batch drawn |
| $n$ | number of batches averaged (`n_batches`) |
| $\hat L$ | the estimate $\frac{1}{n}\sum_i L_i$ |
| $\sigma$ | standard deviation of one batch's loss |
| $\mathrm{Var}$ | variance |
| $\eta_0, \eta_1$ | learning rate at the first and at the last step (`lr`, `lr_final`) |
| $\eta_t$ | learning rate at step $t$ |
| $t$ | step number, $0 \dots S-1$ |
| $S$ | total number of steps (`steps`) |
:::

::: numpy
### Mean of a Python list

```python
losses = [1.5, 2.0, 2.5]
print(np.mean(losses))            # 2.0
print(type(float(np.mean(losses))))   # <class 'float'>   a plain Python number
```

### Powers with a fractional exponent

`a ** 0.5` is the square root of `a`. The decay formula below raises a ratio to the power `step / (steps - 1)`:

```python
print(1e-2 * (1e-4 / 1e-2) ** 0.5)    # 0.001   halfway between 1e-2 and 1e-4, on a log scale
```

### A missing value

`config["lr_final"]` may be `None`, Python's "no value". Test for it with `if config["lr_final"] is None:` (or `if not ...:`).
:::

::: concept
### One batch is not a measurement

The loss of one batch of 128 examples is noisy. For the trained reference model, 500 validation batches gave losses from 1.21 to 2.12, with standard deviation $\sigma \approx 0.15$. Averaging $n$ independent batches divides the variance by $n$:

$$
\mathrm{Var}(\hat L) = \mathrm{Var}\left(\frac{1}{n} \sum_{i=1}^{n} L_i\right) = \frac{1}{n^2} \cdot n\,\sigma^2 = \frac{\sigma^2}{n}
$$

So the estimate's standard deviation is $\frac{\sigma}{\sqrt{n}}$. With $n = 50$: $\frac{0.15}{\sqrt{50}} \approx 0.021$ (measured: 0.023). That is precise enough to compare two runs that differ by 0.05; one batch is not. `estimate_loss` draws $n$ batches with `get_batch` and returns the mean of their losses. The final number in `results.json` uses 200 batches of 512 examples.

### The gap between training and validation loss

Training loss measures what the model has memorized; validation loss measures how well it handles new text. The reference run ends with **training 1.37 and validation 1.68**. When this gap keeps growing while the validation loss stops falling or rises, the model is **overfitting**: its 134K parameters memorize parts of a 1M-character text. The remedies: stop earlier, use a smaller model, or use more data (Stage 3). The opposite case, training and validation loss equal and both still falling, is underfitting: train longer or use a bigger model.

### Choosing a learning rate: the range test

Run short trainings (1,500 Adam steps here) with learning rates spaced evenly on a log scale, and compare the final losses. The reference results, as the mean training loss over the last 100 steps:

| learning rate | $10^{-4}$ | $3 \cdot 10^{-4}$ | $10^{-3}$ | $3 \cdot 10^{-3}$ | $10^{-2}$ | $3 \cdot 10^{-2}$ | $10^{-1}$ |
| --- | --- | --- | --- | --- | --- | --- | --- |
| loss | 2.50 | 2.24 | 2.00 | **1.98** | 2.44 | 4.61 | 15.4 |

Too small crawls. Too large becomes unstable and then blows up. The best short-run value ($3 \cdot 10^{-3}$) is a good **starting** rate. The spacing is on a log scale because learning rates act by multiplication: going from $10^{-3}$ to $3 \cdot 10^{-3}$ is as big a change as going from $10^{-2}$ to $3 \cdot 10^{-2}$.

### Large steps early, small steps late

Early in training, large steps cover ground fast. Late in training, the noise of batch gradients keeps a model with large steps bouncing around a minimum instead of settling into it. So we lower the rate during the run, by the same factor at every step, from $\eta_0$ down to $\eta_1$:

$$
\eta_t = \eta_0 \left(\frac{\eta_1}{\eta_0}\right)^{t/(S-1)}
$$

Check the ends: at $t = 0$ the exponent is 0, which gives $\eta_0$. At $t = S-1$ the exponent is 1, which gives $\eta_0 \cdot \frac{\eta_1}{\eta_0} = \eta_1$. Halfway, the exponent is $\frac{1}{2}$:

$$
\eta_{\text{mid}} = \eta_0 \left(\frac{\eta_1}{\eta_0}\right)^{1/2} = \sqrt{\eta_0\, \eta_1}
$$

This is the geometric mean: $10^{-2}$ and $10^{-4}$ meet at $10^{-3}$.

How much this matters: the default run (Adam, $3 \cdot 10^{-3}$ falling to $3 \cdot 10^{-4}$) ends at validation **1.68**. The same run at a constant $3 \cdot 10^{-3}$ ends at **1.90**.

### results.json

`main()` in `train_mlp.py` is given. It trains with `CONFIG`, then writes `runs/s1/results.json`: the final validation and training loss (200 batches of 512 examples), the time, the number of steps and parameters, the config and a text sample. It also saves the parameters and the vocabulary to `runs/s1/params.npz`. Keeping each run's config next to its result is how you keep track of what you tried.

The reference run reached validation **1.682** in **1.4 minutes** on a 4-core cloud CPU. The boss target is 1.75.
:::

::: think
Q: Your evaluation uses 5 batches. It reports 1.70 for run A and 1.74 for run B. Is A better?
A: You can't tell yet. With $\sigma \approx 0.15$, five batches give an error of about $\frac{0.15}{\sqrt 5} \approx 0.07$, which is larger than the difference. Evaluate again with 50 or more batches (error about 0.02) before you decide.
:::

::: think
Q: The validation loss goes 2.03, 1.89, 1.80, 1.72, 1.70, 1.71, 1.70, while the training loss keeps falling to 1.37. What is happening, and what would you change first?
A: Overfitting: the model keeps improving on the text it has seen, but no longer on new text. Training longer won't help. You could stop earlier (around the minimum), use a smaller hidden layer, or use more data. For the boss target, this curve is already good enough.
:::

::: think
Q: In the range test, why compare losses after a fixed number of steps instead of training each candidate to the end?
A: Training to the end costs a full run per candidate. The short run already shows what the learning rate mostly controls: whether training is stable and how fast it improves. The decay schedule then takes care of the late phase.
:::

::: hint level=1
`estimate_loss` is a short loop: draw a batch, compute its loss, collect it, and return the average. `lr_at` is one formula whose exponent goes from 0 to 1 over the run.
:::

::: hint level=2
Call `get_batch(data, B, k, rng)` exactly `n_batches` times, take `forward(params, x, y)[0]` as each loss, and return `float(np.mean(...))`. In `lr_at`, the fraction is `step / (config["steps"] - 1)`; return `config["lr"]` unchanged when `lr_final` is `None`.
:::

::: hint level=3
```pseudo
estimate_loss(params, data, B, k, n_batches, rng):
    losses ← []
    repeat n_batches times:
        x, y ← get_batch(data, B, k, rng);  append the loss from forward(params, x, y)
    return mean(losses) as a Python float

lr_at(step, config):
    if lr_final is None: return lr
    frac ← step / (steps − 1)
    return lr · (lr_final / lr) ^ frac
```
:::

::: attempt
The editor below opens `workspace/stage1/train_mlp.py`: fill in `estimate_loss` and `lr_at`.
Check: press **Run check** in the Check section below.
Then press **Train the Stage 1 model** in the Run section below (about 1.5 minutes on the reference CPU), and take on the boss in the s1-boss lesson.
:::

::: solution
@include solutions/stage1/train_mlp.py#estimate_loss

- Drawing through `get_batch` with the caller's `rng` makes the estimate repeatable for a fixed seed.
- `float(np.mean(...))` returns a plain number for logs and for the JSON file.

@include solutions/stage1/train_mlp.py#lr_at

- `max(1, steps - 1)` avoids a division by zero in a one-step run. `not lr1` treats `None` (and 0) as "no decay".
:::

::: breakit
Evaluate the same trained model with `n_batches = 1` ten times (with different seeds), and with `n_batches = 50` ten times. The single-batch values spread over about ±0.3 (for example 1.45 to 1.95); the 50-batch values agree within about ±0.05.
:::

::: breakit
Go back to the break-it of s1-l01: split the data after shuffling the windows (so validation windows overlap training ones), train, and compare. The "validation" loss comes out clearly below the honest 1.68, close to the training loss. The leak makes the model look better at new text than it is.
:::

::: recap
- Average many batches: the noise of the estimate falls like $\sigma / \sqrt{n}$.
- The gap between training and validation loss tells overfitting from underfitting.
- A range test with log-spaced learning rates picks the starting rate; lowering it to `lr_final` gains 0.2 here.
- `results.json` records each run's numbers and config.
:::
