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
  threshold: 1.60
  must_beat: runs/s1/results.json
  run: python -m workspace.stage2.train_tf
jobs: [train-s2, resume-s2]
---

::: goal
You will train your own transformer on Tiny Shakespeare until its validation loss is at most **1.60** and lower than your Stage 1 model's, using only the code you wrote.
:::

::: concept
### The run

1. Press **Train the transformer (Stage 2)** in the Run section below. It trains with the `CONFIG` at the top of `workspace/stage2/train_tf.py`, shows the validation loss every 250 steps, saves `runs/s2/ckpt.npz`, and at the end writes `runs/s2/results.json` with a text sample.
2. Press **Run check**. The check reads `results.json`, compares `val_loss` with the target, and compares it with `runs/s1/results.json` from the Stage 1 boss.

If the run stops (a crash, a closed window), press **Continue Stage 2 from its checkpoint**: it starts from the last saved checkpoint.

### The reference, and an estimated target

The reference run was stopped early, at step 1,500 of 5,000, when the course's build machine had to shut down. At that point its validation loss was **1.72** (2.48 bits per character), still falling by about 0.03 every 250 steps, at about 4,000 to 5,000 characters per second on a 4-core cloud CPU. Stage 1 finished at 1.682.

So the target, **1.60**, is an estimate: a full 5,000-step run with the learning rate decaying to its minimum should end clearly below it. If your full run writes good text but ends between 1.60 and 1.68, the estimate was too strict: change `threshold` in this lesson's front matter (`lessons/s2/boss.md`) to your result plus 0.05.

The reference sample at step 1,500 (temperature 0.8). A finished run should look more like Shakespeare than this:

> That with tale's man bloody; pray, forth!
> Commandopatan: that John willing just,
> Now thy susbews with to my broar power his feel,
> Or pensive to my soul and were insteed?
> 
> YORK:
> Your broody.
> 
> KING EDWARD IV:
> And id, housemaberd, sir, the sit would unto
> you their shall day; structs to the's garling;

### If you are above the target

1. **Check correctness first.** Every check from s2-l01 to s2-l09 must pass, and the first logged loss must be near 4.17.
2. **Train longer.** If the validation loss is still falling at the end, raise `max_steps` (the schedule stretches with it).
3. **Learning rate.** Try `lr_max` of `5e-4` or `2e-3`.
4. **Size.** `n_layers = 6` or `d = 192` learn more per step, but each step is slower.
:::

::: attempt
Train: press **Train the transformer (Stage 2)** in the Run section below.
Check: press **Run check** when the run has finished.
:::

::: recap
- You trained a transformer, written on your own autograd engine, end to end.
- Attention over 128 characters beats the 8-character MLP of Stage 1.
- Next, Stage 3 replaces characters with subword tokens and scales up.
:::
