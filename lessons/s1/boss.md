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
jobs: [train-s1, sample-s1]
---

::: goal
You will train your own Stage 1 model on Tiny Shakespeare until its validation loss is at most **1.75**, using only the code you wrote in s1-l01 to s1-l13.
:::

::: concept
### The run

From the repository root:

1. Press **Train the Stage 1 model** in the Run section below. It trains with the `CONFIG` at the top of `workspace/stage1/train_mlp.py` and writes `runs/s1/results.json`, `runs/s1/params.npz` and a sample.
2. Press **Run check**. It reads `results.json` and compares `val_loss` with the target.

`estimate_loss` measures `val_loss` at the end, on 200 validation batches of 512 examples. So it is a stable number, not one lucky batch.

### The reference

The reference solution, with the default config (40,000 Adam steps, $B = 128$, learning rate decaying from $3 \times 10^{-3}$ to $3 \times 10^{-4}$), reached **1.682** validation loss (train loss 1.369) in **1.4 minutes** on a 4-core cloud CPU. The target, 1.75, leaves some room for a different machine and seed.

For comparison: guessing uniformly scores $\ln 65 = 4.17$. Predicting each character from how often it occurs, without looking at the context, scores about 3.3. Getting below 2 takes the previous 8 characters.

### If you are above the target

Work through this list in order, and change one thing at a time:

1. **Check correctness first.** Every check from s1-l01 to s1-l13 must pass, and the loss at step 0 must be near 4.17.
2. **Look at the curve.** If the training loss is still falling at the end, train longer (more `steps`). If the validation loss rises while the training loss keeps falling, the model is overfitting: stop earlier or make the model smaller.
3. **Learning rate.** Repeat the range test of s1-l12 for your config, and pick the largest rate whose curve is still smooth.
4. **Model size.** A longer context (`k = 10`) or larger embeddings (`e = 32`) help a little. A larger hidden layer mostly overfits faster on this small text.

Write down what you tried. Real models are tuned with exactly such a log.
:::

::: attempt
Train: press **Train the Stage 1 model** in the Run section below.
Check: press **Run check** in the Check section below.
:::

::: recap
- You trained a neural language model with code you wrote yourself, from the vocabulary to the optimizer.
- You can read a training and validation curve and know which setting to change.
- Next, Interlude A builds a tool that computes the backward pass for you.
:::
