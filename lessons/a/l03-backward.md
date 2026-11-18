---
id: a-l03
title: "backward(): topological sort"
stage: a
order: 3
prereqs: [a-l02]
files: [workspace/autograd/tensor.py]
functions: [backward]
tests: tests/a/test_l03_backward.py
est_minutes: 45
---

::: goal
Write `backward(root)`: order every tensor in the graph so each one comes after all the tensors that use it in the backward direction, seed $\frac{\partial L}{\partial L} = 1$, and run every `_backward` in that order, without recursion.
:::

::: notation
| Symbol | Meaning |
| --- | --- |
| root | the scalar tensor we differentiate (the loss $L$) |
| $a, b, c, d$ | tensors in an example graph |
| parent | an input of an operation (an entry of `_prev`) |
| topological order | a list where every tensor appears after all of its parents |
| DFS | depth-first search |
| leaf | a tensor with no parents (parameter or input) |
| $L$ | the loss |
| $x$ | a leaf in the chain example |
:::

::: concept
### Why order matters

A node's `_backward` uses its own `grad` and pushes into its parents. That is only correct once its own `grad` is **complete**, i.e. after every node that uses it has already pushed its contribution.

The diamond: $b = 2a$, $c = 3a$, $d = b \cdot c = 6a^2$, so $\frac{\partial d}{\partial a} = 12a$, which is 24 at $a = 2$.

| order of `_backward` calls | `a.grad` |
| --- | --- |
| `d`, `b`, `c`, then `a` (correct) | $2 \cdot c + 3 \cdot b = 2 \cdot 6 + 3 \cdot 4 = 24$ |
| `d`, `b`, `a`, `c` (wrong: `a` runs before `c` is done) | leaves are fine here, but if `a` were an intermediate node it would pass on only $12$ |

The rule: **run `_backward` on a node only after every node that consumes it has run.** That is exactly the reverse of a topological order.

### Topological order by depth-first search

Visit the root; before placing a node in the list, visit (and place) all its parents. A node is appended only after all its parents, so the list is topologically sorted: parents first, root last. Reversed, it starts at the root and every node comes after all of its consumers.

For the diamond, one valid order is `[a, b, c, d]`; reversed: `d, c, b, a`.

A `seen` set makes each node appear once, even when it is reachable along several paths (`a` is reachable through `b` and through `c`). Without it, `a._backward` would run twice and double-count.

### No recursion

The natural recursive version (`visit(node)` calls `visit(parent)`) needs one Python stack frame per level of depth. Python's default limit is 1,000 frames. A transformer's graph is not that deep, but a long chain of operations is: the check builds a chain of 10,000 nodes. Replace the call stack with an explicit list used as a stack. Push `(node, expanded=False)`; when popped unexpanded, mark it seen, push it back as `(node, True)` and push its unseen parents; when popped expanded, all its parents have been placed, so append it.

### Seeding and resetting

- The root gets `grad = 1` (the derivative of $L$ with respect to itself), shaped like its data.
- **Leaves accumulate**: parameters keep adding gradients across calls, which is why training calls `zero_grad()` on every parameter before each step. This is the same `+=` that handles a node used twice.
- **Intermediate nodes are reset** to zero at the start of every `backward`, so calling `backward` twice gives leaves exactly twice the gradient, never a muddle of stale intermediate values.
:::

::: think
Q: Why isn't it enough to call `_backward` in the reverse of the order in which operations were *created*?
A: In a single forward pass, creation order is in fact a topological order, so it would work, if you recorded it somewhere global. DFS from the root needs no global record, handles graphs built in any order, and only visits nodes that actually influence the root.
:::

::: think
Q: `a` is reachable from the root through two different paths. What goes wrong without the `seen` set?
A: `a` appears twice in the order, so `a._backward()` runs twice and pushes its gradient into its own parents twice, doubling their gradients (and the work grows exponentially in graphs with many such diamonds).
:::

::: think
Q: You forget to zero the parameters' gradients between two training steps. What does the second step's update use?
A: The sum of both steps' gradients: step 2 moves the parameters as if the first batch's gradient were still present. Over many steps gradients keep growing, and the effective learning rate explodes.
:::

::: hint level=1
Two phases: build the order with an explicit stack, then walk it backwards calling `_backward`. Between them, reset intermediate grads and seed the root.
:::

::: hint level=2
Use a Python list as the stack and a `set` of `id(node)` for the visited nodes (`id` because Tensors are not hashable by value). Push `(node, False)` to mean "visit" and `(node, True)` to mean "all parents are placed, append me". `np.ones_like(root.data)` seeds the root; `reversed(topo)` walks backwards.
:::

::: hint level=3
```pseudo
backward(root):
    topo ← [];  seen ← empty set;  stack ← [(root, false)]
    while stack is not empty:
        node, expanded ← pop
        if expanded: append node to topo;  continue
        if node in seen: continue
        add node to seen;  push (node, true)
        for each parent of node not yet seen: push (parent, false)
    for each node in topo that has parents: node.grad ← zeros
    root.grad ← ones like root.data
    for node in reverse(topo): node._backward()
```
:::

::: attempt
File: `workspace/autograd/tensor.py` — implement the module-level `backward(root)` (`Tensor.backward()` already calls it).
Check: `python -m course check a-l03`
:::

::: solution
@include solutions/autograd/tensor.py#backward

- The `(node, expanded)` pairs emulate recursion: a node is appended only when popped the second time, after all of its parents were pushed later and therefore popped (and placed) first.
- `seen` uses `id(node)`, so each node is placed once however many paths lead to it.
- Intermediate grads are reset each call; leaves are not, so gradient accumulation across calls is explicit and controlled by `zero_grad`.
- `reversed(topo)` starts at the root: every node's grad is complete before its `_backward` runs.
:::

::: breakit
Write the recursive version (`def visit(n): for p in n._prev: visit(p); topo.append(n)`) and run it on the check's 10,000-node chain. Expect `RecursionError: maximum recursion depth exceeded` after about 1,000 levels.
:::

::: breakit
Remove the `seen` check and run the diamond. Expect `a.grad` larger than 24 (nodes reachable twice get processed twice).
:::

::: recap
- `_backward` must run on a node only after all its consumers: the reverse of a topological order.
- DFS with an explicit stack builds that order without hitting Python's recursion limit.
- The root is seeded with 1; intermediates are reset each call; leaves accumulate until `zero_grad`.
:::
