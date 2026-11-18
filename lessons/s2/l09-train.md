---
id: s2-l09
title: Train, checkpoint, sample
stage: s2
order: 9
prereqs: [s2-l08]
files: [workspace/stage2/model.py, workspace/stage2/train_tf.py]
functions: [save_checkpoint, load_checkpoint, generate, train]
tests: tests/s2/test_l09_train.py
est_minutes: 60
jobs: [train-s2, resume-s2]
---

::: goal
You will train the transformer on Tiny Shakespeare. You write the training loop, save and load the model's weights (checkpoints), and let the model write text. You also compare it with Stage 1 in **bits per character**.
:::

::: notation
| Symbol | Meaning |
| --- | --- |
| $L$ | mean cross-entropy loss, in nats (natural log) |
| bpc | bits per character: the loss measured with $\log_2$ instead of $\ln$ |
| $\ln 2$ | $\approx 0.693$, the factor between nats and bits |
| $T$ | context length (128) |
| $t$ | the length of the current input during generation |
:::

::: numpy
### Saving and loading arrays by name

`np.savez(path, name=array, ...)` writes several arrays into one `.npz` file. `np.load(path)` reads them back:

```python
np.savez("demo.npz", W=np.ones((2, 2)), b=np.zeros(2))
saved = np.load("demo.npz")
print(saved.files, saved["W"].shape)      # ['W', 'b'] (2, 2)
```

To save a dictionary of arrays, unpack it with `**`: `np.savez(path, **{name: array, ...})`.

### Copying into an existing array

`p[...] = new_values` overwrites the contents of `p` and keeps the same array object:

```python
p = np.zeros((2, 2))
ref = p
p[...] = saved["W"]
print(ref, ref is p)
# [[1. 1.]
#  [1. 1.]] True                          the optimizer, which holds `ref`, sees the new values
```

### Freeing memory

Every Tensor's backward function refers back to the Tensor itself, so each graph is a "reference cycle". Python frees cycles only now and then, and a graph of this model holds hundreds of MB. After each step, `del loss` and `gc.collect()` (from `import gc`) free the graph right away.

### The last `T` items of a list

```python
ids = [1, 2, 3, 4, 5]
print(ids[-3:])                           # [3, 4, 5]
```
:::

::: concept
### The training loop

The loop is the one from s1-l10, with the new tools from s2-l08:

1. set the learning rate from the schedule;
2. draw a batch with `get_batch_seq`;
3. zero the gradients, run the model, run `loss.backward()`;
4. clip the gradients;
5. take an AdamW step;
6. record the loss, then free the graph with `del loss` and `gc.collect()`.

Every `eval_every` steps, `estimate_loss` (given) measures the validation loss, and the loop saves a checkpoint.

The default config: $d = 128$, 4 heads, 4 blocks, $T = 128$, batches of 32 sequences, 5,000 steps. One step processes $32 \cdot 128 = 4{,}096$ characters. The reference CPU managed about 5,300 characters per second, so a full run takes a little over an hour. Expect one to two hours on a typical desktop.

### Checkpoints

A long run can be interrupted: a crash, a power cut, a closed laptop. A checkpoint saves every parameter by its name from `named_parameters()`, for example `blocks.0.attn.qkv.W`. Loading copies the saved arrays back **into the existing Tensors** (`p.data[...] = saved[name]`), so the optimizer, which holds those Tensors, keeps working. Loading checks every name and shape, so a checkpoint from a different model size fails loudly instead of loading garbage. The **Continue Stage 2 from its checkpoint** button (in the Run section) continues from `runs/s2/ckpt.npz`.

(For a perfect resume you would also save AdamW's `m`, `v` and `t`, and the step number. We keep it simple: resuming restarts the schedule with the saved weights.)

### Generating text

To write text, the model reads the ids so far and predicts the next one from the logits at the **last** position: `logits[0, -1]`. The model only has position vectors for $T$ positions, so feed at most the last $T$ ids. Then draw the next id with your Stage 1 `sample_next`, append it, and repeat.

The causal mask means that every earlier position's prediction is computed too, and thrown away. That wastes work, but keeps the code simple.

### Bits per character: comparing with Stage 1

Our losses are in **nats**, because cross-entropy uses the natural log. Changing the base of a logarithm divides by a constant, $\log_2 x = \frac{\ln x}{\ln 2}$, so

$$
\text{bpc} = \frac{L}{\ln 2}
$$

Stage 1 reached $L = 1.682$, which is $\frac{1.682}{0.693} = 2.43$ bits per character. In words: on average the MLP narrows the next character down to about $2^{2.43} \approx 5.4$ equally likely options. Bits per character will matter in Stage 3, where a token is no longer one character (s3-l04).
:::

::: think
Q: Why must `load_checkpoint` copy into the existing arrays instead of replacing `p.data` with the loaded array?
A: Both work for the model itself. The optimizer, though, was built before loading and keeps its own `m` and `v` per parameter name, matched to the existing Tensors. Copying in place keeps every reference consistent; it also catches shape mismatches, because NumPy refuses to copy a `(16, 48)` array into a `(8, 24)` one.
:::

::: think
Q: A loss of 1.5 nats: how many bits per character is that, and how many equally likely choices does it correspond to?
A: $\frac{1.5}{0.693} = 2.16$ bits, so about $2^{2.16} \approx 4.5$ equally likely next characters.
:::

::: think
Q: During generation, why use the logits at position $-1$ and not at position 0?
A: Position $t$ predicts the character after position $t$. The last position has seen the whole context and predicts the character that comes next. Position 0 has seen only the first character.
:::

::: hint level=1
`save_checkpoint` and `load_checkpoint` work with `model.named_parameters()`. `generate` is a loop: crop to $T$ ids, run the model, take the last position's logits, draw, append. `train` follows the six-step loop above and its docstring.
:::

::: hint level=2
Save: `np.savez(path, **{name: p.data for name, p in params.items()})`. Load: `np.load(path)`, check `set(params) - set(saved.files)` for missing names and compare shapes, then `p.data[...] = saved[name]`. Generate: `np.array([ids[-model.T:]], dtype=np.int64)`, `logits.data[0, -1]`, `sample_next(...)`. Train: copy the structure of Stage 1's `train`, with `clip_grad_norm` and `AdamW`.
:::

::: hint level=3
```pseudo
generate(model, ids, n, temperature, rng):
    repeat n times:
        context ← the last T ids, as a (1, t) array
        logits, _ ← model(context)
        append sample_next(logits at batch 0, last position, temperature, rng) to ids
        free the graph (gc.collect)
    return ids

train(config, train_data, val_data, V):
    model ← GPT(...);  load the checkpoint if resuming;  opt ← AdamW(model.named_parameters(), ...)
    for each step:
        lr ← lr_at(step, ...);  x, y ← get_batch_seq(train_data, B, T, rng)
        model.zero_grad();  _, loss ← model(x, y);  loss.backward()
        clip_grad_norm(model.parameters(), grad_clip);  opt.lr ← lr;  opt.step()
        record the loss;  delete loss;  gc.collect()
        every eval_every steps: estimate the validation loss, log it, save a checkpoint
    return {model, train_loss, val_loss}
```
:::

::: attempt
The editor below has a tab for each file. In `workspace/stage2/model.py`: fill in `save_checkpoint` and `load_checkpoint`; in `workspace/stage2/train_tf.py`, fill in `generate` and `train`.
Check: press **Run check** in the Check section below.
Then press **Train the transformer (Stage 2)** in the Run section below (one to two hours), and take on the boss in the s2-boss lesson.
:::

::: solution
@include solutions/stage2/model.py#save_checkpoint

@include solutions/stage2/model.py#load_checkpoint

- Missing names and wrong shapes raise clear errors instead of loading a broken model.
- `p.data[...] = saved[name]` keeps the same Tensor objects, so the optimizer's references stay valid.

@include solutions/stage2/train_tf.py#generate

- `ids[-model.T:]` handles prompts both shorter and longer than $T$.
- `logits.data[0, -1]` is the prediction for the next character.

@include solutions/stage2/train_tf.py#train

- The loop matches the six steps above; the checkpoint is saved at every evaluation, so an interrupted run loses at most `eval_every` steps.
- `del loss` and `gc.collect()` keep memory at about 3 GB instead of growing without limit.
:::

::: breakit
Remove the `gc.collect()` line from `train` and watch the memory use of the Python process (Task Manager on Windows, `top` on Linux) during the first 100 steps. It grows by hundreds of MB, and a long run eventually crashes. Put the line back and memory stays flat.
:::

::: recap
- The training loop adds the schedule, gradient clipping and AdamW to the s1-l10 loop, and frees each step's graph.
- Checkpoints save parameters by name and load them back in place, with name and shape checks.
- Generation feeds at most the last $T$ ids and samples from the last position's logits.
- bpc $= L / \ln 2$; Stage 1 scored 2.43 bits per character.
:::
