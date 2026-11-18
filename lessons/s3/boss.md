---
id: s3-boss
title: "Boss: write short stories"
stage: s3
order: 99
kind: boss
prereqs: [s3-l05]
est_minutes: 480
boss:
  results: runs/s3/results.json
  metric: val_bpc
  threshold: 1.5
  ask: "Read the sample above. Is it a coherent short story (sentences that make sense together)?"
  run: python -m workspace.stage3.train_bpe_lm
jobs: [data-tinystories, train-s3, resume-s3]
---

::: goal
You will train the subword transformer on TinyStories, overnight, until it writes coherent short stories and scores at most **1.5** bits per character on held-out stories.
:::

::: concept
### The run

1. Press **Download TinyStories** in the Run section below (about 120 MB, once).
2. Press **Train the BPE model (Stage 3)**. It learns the BPE merges, tokenizes the data (a few minutes, once), and trains, saving a checkpoint every 500 steps. If it stops, press **Continue Stage 3 from its checkpoint**.
3. When it finishes, read the sample story the check shows, answer the question with **Yes** or **No** in the Check section, and press **Run check**. The check reads `runs/s3/results.json` and compares `val_bpc` with the target.

### The target is an estimate

The other bosses were calibrated with a reference run on the course's build machine. This one could not be: a full Stage 3 run takes a night, and that machine could not stay up that long. So the target, **1.5 bits per character**, is an estimate. It is loose on purpose, well below the 8 bits of a uniform byte model and the 2.43 bits of Stage 1's character model on Shakespeare. A model that writes readable stories should beat it.

After your run, look at your own number and your sample. If your model clearly writes stories but misses 1.5, the estimate was too strict: change `threshold` in this lesson's front matter (`lessons/s3/boss.md`) to your result plus 0.05, and note what you changed. If it beats 1.5 easily, try lowering the target to your result plus 0.05 and train a second, better run.

### If you are above the target

1. **Check correctness first.** All Stage 2 and Stage 3 checks must pass.
2. **More tokens.** Train longer (`max_steps`), within the time you have; use `estimate_hours`.
3. **Learning rate.** `lr_max` between `3e-4` and `1e-3`.
4. **Size.** With little time, a smaller model (`d = 192`) can win; with more time, a larger one.

### Your projects

Your Korean morphological analyzer worked on jamo. Try BPE on Korean text: train merges on a Korean corpus three ways (on UTF-8 bytes as here, on whole syllables, and on jamo after decomposing each syllable) and compare how many tokens the same sentences need, and which pieces the merges discover. Bytes always work; jamo merges often rediscover morphemes.
:::

::: attempt
Train: press **Train the BPE model (Stage 3)** in the Run section below (overnight).
Check: press **Run check** in the Check section below.
:::

::: recap
- You built a complete language model pipeline from scratch: tokenizer, data on disk, autograd engine, transformer, optimizer, schedule, evaluation and sampling.
- You can plan a training run from its compute budget, and compare models across tokenizers in bits per character.
:::
