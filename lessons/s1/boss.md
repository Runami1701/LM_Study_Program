---
id: s1-boss
title: "Boss: train the MLP to target"
stage: s1
order: 99
kind: boss
prereqs: [s1-l13]
est_minutes: 60
boss:
  results: runs/s1/results.json
  metric: val_loss
  threshold: 1.75
  run: python -m workspace.stage1.train_mlp
---

::: goal
Train your own stage-1 model on Tiny Shakespeare until its validation loss is at most **1.75** nats per character, using only the code you wrote in s1-l01 to s1-l13.
:::

::: concept
### The run

From the repository root:

1. `python -m workspace.stage1.train_mlp` trains with the `CONFIG` at the top of the file and writes `runs/s1/results.json`, `runs/s1/params.npz` and a sample.
2. `python -m course check s1-boss` reads `results.json` and compares `val_loss` with the target.

`val_loss` is measured at the end by `estimate_loss` on 200 validation batches of 512 examples, so it is a stable number, not one lucky batch.

### The reference

The reference solution, with the default config (40,000 Adam steps, $B = 128$, learning rate decaying from $3 \times 10^{-3}$ to $3 \times 10^{-4}$), reached **1.682** validation loss (train loss 1.369) in **1.4 minutes** on a 4-core cloud CPU. The target, 1.75, leaves some room for a different machine and seed.

For scale: uniform guessing scores $\ln 65 = 4.17$; predicting each character from its frequency alone (no context) scores about 3.3; this model needs the previous 8 characters to get below 2.

### If you are above the target

Work through these in order and change one thing at a time:

1. **Check correctness first.** All of s1-l01 to s1-l13 must pass, and the loss at step 0 must be near 4.17.
2. **Look at the curve.** If training loss is still falling at the end, train longer (more `steps`). If validation loss rises while training loss keeps falling, you are overfitting: stop earlier or make the model smaller.
3. **Learning rate.** Redo the range test of s1-l12 for your config; pick the largest rate whose curve is still smooth.
4. **Capacity.** A larger context (`k = 10`) or embedding (`e = 32`) helps a little; a larger hidden layer mostly overfits faster on this small text.

Record what you tried. That log is how real models get tuned.
:::

::: attempt
Train: `python -m workspace.stage1.train_mlp`
Check: `python -m course check s1-boss`
:::

::: recap
- You trained a neural language model end to end from code you wrote yourself, from the vocabulary to the optimizer.
- You can read a train/validation curve and know which knob to turn.
- Next, Interlude A automates the backward pass you have been writing by hand.
:::
