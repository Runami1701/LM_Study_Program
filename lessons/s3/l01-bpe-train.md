---
id: s3-l01
title: Training BPE
stage: s3
order: 1
prereqs: [s2-boss]
files: [workspace/stage3/bpe.py]
functions: [merge_pair, count_pairs, train_bpe]
tests: tests/s3/test_l01_bpe_train.py
est_minutes: 60
---

::: goal
You will build a **subword** vocabulary with byte-pair encoding (BPE). BPE starts from the 256 possible bytes and repeatedly merges the most frequent neighbouring pair into a new token, until the vocabulary has `vocab_size` tokens. Frequent words then become a single token, and rare words are spelled with a few pieces.
:::

::: notation
| Symbol | Meaning |
| --- | --- |
| byte | a whole number from 0 to 255; UTF-8 writes every character as 1 to 4 bytes |
| token | one entry of the vocabulary: a byte (ids 0–255) or a merged sequence of bytes (ids 256 and up) |
| pair | two neighbouring token ids, written $(a, b)$ |
| merge | the rule "replace the pair $(a, b)$ by the new id $n$" |
| $V$ | vocabulary size (`vocab_size`), 4096 in our run |
| $256 + k$ | the id created by merge number $k$ ($k = 0, 1, \dots$) |
| chunk | a piece of text from `pretokenize`: a word with its leading space, a number, punctuation, or whitespace |
| $a, b, c, d$ | in the worked example: the letters a, b, c, d (bytes 97 to 100) |
| $Z, Y, X$ | in the worked example: the new tokens 256, 257, 258 |
:::

::: numpy
### Text to bytes and back

`str.encode("utf-8")` turns text into bytes. `list(...)` gives the byte values as numbers. Korean characters take 3 bytes each:

```python
print(list("hi 관".encode("utf-8")))          # [104, 105, 32, 234, 180, 128]
print(bytes([104, 105]))                       # b'hi'
print(bytes([234, 180, 128]).decode("utf-8"))  # 관
```

### Neighbouring pairs

`zip(ids, ids[1:])` pairs each id with the next one:

```python
ids = [97, 97, 97, 98]
print(list(zip(ids, ids[1:])))      # [(97, 97), (97, 97), (97, 98)]
```

### Counting with Counter

`Counter` is a dictionary that starts every count at 0:

```python
from collections import Counter
c = Counter()
c[(97, 97)] += 2
c[(97, 98)] += 1
print(c[(1, 2)])                    # 0   a pair never seen counts as 0
print(Counter(["ab", "ab", "cd"]))  # Counter({'ab': 2, 'cd': 1})
```

### The most frequent pair, smallest on a tie

`min` with a `key` function picks the item with the smallest key. Sorting by `(-count, pair)` puts the largest count first, and on equal counts the smallest pair:

```python
pairs = {(97, 98): 2, (256, 97): 2, (98, 99): 1}
print(min(pairs, key=lambda p: (-pairs[p], p)))   # (97, 98)
```

Dictionary keys must not change, so a chunk's ids are stored as a **tuple** (which cannot change), not a list.
:::

::: concept
### Why not characters, and why not whole words?

Stage 2 used one token per character. That works, but every story becomes a long sequence: the model spends its $T$ positions and its computation on single letters. Whole words go too far the other way: a vocabulary of every word is huge, and an unknown word cannot be written at all.

Subwords sit in between. Common words ("the", " was", " happy") become single tokens. Rare words are spelled with a few common pieces (" un", "believ", "able"). And because the vocabulary starts from all 256 bytes, **any** text can be written, including Korean, emoji, or a typo.

### The algorithm

1. Write the text as bytes. The vocabulary starts as the 256 byte values.
2. Count every pair of neighbouring tokens.
3. Take the most frequent pair $(a, b)$. Give it the next free id, $256 + k$, and replace every occurrence of $(a, b)$ by that id.
4. Repeat steps 2 and 3 until the vocabulary has $V$ tokens, that is, $V - 256$ merges.

The result is the ordered **list of merges**. That list is the whole tokenizer: lesson s3-l02 applies it to new text.

### Worked example: `"aaabdaaabac"`

The bytes are `a a a b d a a a b a c` (a = 97, b = 98, c = 99, d = 100). Write $Z, Y, X$ for the new ids 256, 257, 258.

**Merge 1.** Pair counts: `aa` 4 (overlaps count: `aaa` contains two `aa` pairs), `ab` 2, `bd` 1, `da` 1, `ba` 1, `ac` 1. Merge `aa` → $Z$. Replace from left to right without overlap, so `aaa` becomes $Z$ `a`:

$$
Z\ a\ b\ d\ Z\ a\ b\ a\ c
$$

**Merge 2.** Counts: $(Z, a)$ 2, $(a, b)$ 2, all others 1. There is a tie. We break ties by taking the **smallest** pair, and $(97, 98) < (256, 97)$, so merge `ab` → $Y$:

$$
Z\ Y\ d\ Z\ Y\ a\ c
$$

**Merge 3.** $(Z, Y)$ appears twice; everything else once. Merge it → $X$:

$$
X\ d\ X\ a\ c
$$

The merges are $[(97, 97), (97, 98), (256, 257)]$: 11 bytes became 5 tokens. The tie rule is not important for quality, but it must be fixed, so that everyone gets the same tokenizer from the same text. The check tests exactly these three merges.

### Chunks: merges stay inside words

If we merged across all of the text, BPE would learn tokens like `"e t"` or `". The"`, which glue the end of one word to the start of the next. We prevent this by first splitting the text into **chunks** with a regular expression (`pretokenize`, given): a word together with its leading space (`" happy"`), a number, a run of punctuation, or whitespace. Merges never cross a chunk boundary.

This also gives a big speed-up. TinyStories repeats the same few thousand words over and over. So we count each distinct chunk once, together with how often it occurs: `words = {tuple of bytes: count}`. A pair inside a chunk that occurs 5,000 times counts 5,000 times, but is processed only once.

### Cost

Each merge recounts the pairs of every distinct chunk, and then rewrites the chunks that contain the chosen pair. With about 30,000 distinct chunks and 3,840 merges, this takes a few minutes in plain Python on 10 MB of text. That is why we learn the merges from a 10 MB sample, not from all 100 MB: the most frequent pairs are already clear in the sample.
:::

::: think
Q: Why does `"aaa"` contain two `aa` pairs when counting, but turn into only one $Z$ (plus an `a`) when merging?
A: Counting looks at every neighbouring position: positions (0,1) and (1,2) are both `aa`. Merging replaces non-overlapping occurrences from left to right: after merging (0,1), position 1 is used up, so (1,2) can't merge too. The count is an estimate of how useful the merge is; the merge itself must give a valid sequence.
:::

::: think
Q: Can BPE ever fail to encode a piece of text, for example Korean or an emoji, if the training text had none?
A: No. Every text is a sequence of bytes, and all 256 bytes are in the vocabulary from the start. Text the merges never saw is simply spelled with more, smaller tokens.
:::

::: think
Q: What would happen to the vocabulary without the chunk split?
A: Frequent combinations across word boundaries, such as `"e "` + `"t"` or `". "` + `"T"`, would become tokens, wasting vocabulary slots on fragments that mean nothing. Splitting into chunks keeps every token inside one word, number or punctuation run.
:::

::: hint level=1
`merge_pair` walks through the list once, looking at the current id and the next one. `count_pairs` adds each chunk's count to every pair inside it. `train_bpe` is a loop of: count, pick the best pair, record it, rewrite every chunk.
:::

::: hint level=2
`merge_pair`: a `while i < len(ids)` loop; when `ids[i], ids[i+1]` equals the pair, append `new_id` and advance by 2, otherwise append `ids[i]` and advance by 1. `count_pairs`: `Counter`, and `zip(ids, ids[1:])` for each chunk. `train_bpe`: `Counter(tuple(chunk.encode("utf-8")) for chunk in pretokenize(text))`, then pick with `min(counts, key=lambda p: (-counts[p], p))`, and rebuild `words` with `tuple(merge_pair(...))`.
:::

::: hint level=3
```pseudo
merge_pair(ids, pair, new_id):
    out ← [];  i ← 0
    while i < length of ids:
        if ids[i], ids[i+1] equal pair: append new_id;  i ← i + 2
        else: append ids[i];  i ← i + 1
    return out

train_bpe(text, vocab_size):
    words ← count of each chunk of pretokenize(text), as a tuple of its UTF-8 bytes
    merges ← []
    for new_id from 256 to vocab_size − 1:
        counts ← count_pairs(words);  stop if it is empty
        best ← the pair with the highest count (smallest pair on a tie)
        append best to merges
        words ← every chunk with best replaced by new_id (counts kept)
    return merges
```
:::

::: attempt
File: `workspace/stage3/bpe.py` — fill in `merge_pair`, `count_pairs` and `train_bpe`.
Check: `python -m course check s3-l01`
:::

::: solution
@include solutions/stage3/bpe.py#merge_pair

- `i + 1 < len(ids)` guards the last position, which has no right neighbour.
- Skipping 2 after a merge is what prevents overlapping merges.

@include solutions/stage3/bpe.py#count_pairs

@include solutions/stage3/bpe.py#train_bpe

- `words` counts each distinct chunk once, so frequent words cost no extra work.
- `min(counts, key=lambda p: (-counts[p], p))` is "highest count, then smallest pair".
- `if best[0] in ids` skips chunks that can't contain the pair, which saves most of the rewriting.
- The loop stops early when every chunk is a single token.
:::

::: breakit
Train merges on the first 1 MB of TinyStories with `vocab_size=300` and print the 44 new tokens with `build_vocab(merges)[256:]`. You see common pieces such as `b' t'`, `b'he'`, `b' the'`, `b'ed'` and `b' and'`. Then remove the chunk split (use `[text]` instead of `pretokenize(text)`) and train again on the first 300,000 characters: now tokens that end in a space appear, such as `b'e '`, `b'd '`, `b'the '`, and even `b'. T'`, which glues the end of one sentence to the start of the next.
:::

::: recap
- BPE starts from 256 bytes and repeatedly merges the most frequent neighbouring pair into a new token.
- The ordered list of merges is the tokenizer; any text can be encoded, because all bytes are tokens.
- Chunks keep merges inside words and make training fast; ties go to the smallest pair.
:::
