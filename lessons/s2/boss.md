---
id: s2-boss
title: "Boss: train the transformer to target"
stage: s2
order: 99
kind: boss
prereqs: [s2-l09]
est_minutes: 120
boss:
  results: runs/s2/results.json
  metric: val_loss
  threshold: S2_THRESHOLD
  must_beat: runs/s1/results.json
  run: python -m workspace.stage2.train_tf
---

::: goal
You will train your own transformer on Tiny Shakespeare until its validation loss is at most **S2_THRESHOLD** and lower than your Stage 1 model's, using only the code you wrote.
:::

::: concept
### The run

1. `python -m workspace.stage2.train_tf` trains with the `CONFIG` at the top of the file. It prints the validation loss every 250 steps, saves `runs/s2/ckpt.npz`, and at the end writes `runs/s2/results.json` with a text sample.
2. Press **Run check** (or run `python -m course check s2-boss`). The check reads `results.json`, compares `val_loss` with the target, and compares it with `runs/s1/results.json` from the Stage 1 boss.

If the run is interrupted, set `CONFIG["resume"] = True` and start it again: it continues from the last checkpoint.

### The reference

The reference solution, with the default config (5,000 steps, batches of $32 \times 128$ characters), reached validation loss **S2_REF** (S2_BPC bits per character) in **S2_MIN** minutes on a 4-core cloud CPU, at about S2_TOKS characters per second. Stage 1 reached 1.682. The target, S2_THRESHOLD, leaves room for a different machine and seed.

The reference sample:

S2_SAMPLE

### If you are above the target

1. **Check correctness first.** Every check from s2-l01 to s2-l09 must pass, and the first logged loss must be near 4.17.
2. **Train longer.** If the validation loss is still falling at the end, raise `max_steps` (the schedule stretches with it).
3. **Learning rate.** Try `lr_max` of `5e-4` or `2e-3`.
4. **Size.** `n_layers = 6` or `d = 192` learn more per step, but each step is slower.
:::

::: attempt
Train: `python -m workspace.stage2.train_tf`
Check: press **Run check**, or run `python -m course check s2-boss`
:::

::: recap
- You trained a transformer, written on your own autograd engine, end to end.
- Attention over 128 characters beats the 8-character MLP of Stage 1.
- Next, Stage 3 replaces characters with subword tokens and scales up.
:::
