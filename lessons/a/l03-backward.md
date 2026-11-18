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
You will write `backward(root)`. It puts every tensor of the graph in an order where each tensor comes after all the tensors that use it, sets $\frac{\partial L}{\partial L} = 1$, and calls every `_backward` in that order. It must work without recursion.
:::

::: notation
| Symbol | Meaning |
| --- | --- |
| root | the final tensor we differentiate (the loss $L$) |
| $a, b, c, d$ | tensors in an example graph |
| parent | an input of an operation (an entry of `_prev`) |
| topological order | a list in which every tensor comes after all of its parents |
| leaf | a tensor with no parents (a parameter or an input) |
| $L$ | the loss |
| $x$ | a leaf in the chain example |
:::

::: numpy
### A list used as a stack

`append` adds an item at the end; `pop` removes and returns the last item. So the last item in is the first one out:

```python
stack = [1]
stack.append(2)
stack.append(3)
print(stack.pop(), stack)       # 3 [1, 2]
```

### Remembering which objects you have seen

`id(obj)` is a number that identifies one object. A `set` of such numbers answers "have I seen this object?" quickly:

```python
a, b = [0.0], [0.0]             # two different objects with equal contents
seen = set()
seen.add(id(a))
print(id(a) in seen, id(b) in seen)   # True False
```

We use `id` because two different Tensors can hold equal numbers but are still different nodes.

### Walking a list backwards

`for node in reversed(topo):` visits the list from the last item to the first.
:::

::: concept
### Why the order matters

A node's `_backward` uses its own `grad` and adds into its parents. That is only correct once its own `grad` is **complete**: every node that uses it must already have added its share.

Take the diamond: $b = 2a$, $c = 3a$, $d = b \cdot c = 6a^2$. Then $\frac{\partial d}{\partial a} = 12a$, which is 24 at $a = 2$.

| order of `_backward` calls | `a.grad` |
| --- | --- |
| `d`, `b`, `c`, then `a` (correct) | $2 \cdot c + 3 \cdot b = 2 \cdot 6 + 3 \cdot 4 = 24$ |
| `d`, `b`, `a`, `c` (wrong: `a` runs before `c` has added its share) | still 24 here, because `a` is a leaf; but an intermediate node in `a`'s place would pass on only 12 to its parents |

The rule: **call `_backward` on a node only after every node that uses it has run.** That is exactly the reverse of a topological order.

### A topological order by depth-first search

Visit the root. Before adding a node to the list, visit (and add) all its parents. Since a node is added only after all its parents, the list is in topological order: parents first, root last. Reversed, it starts at the root, and every node comes after all the nodes that use it.

For the diamond, one valid order is `[a, b, c, d]`; reversed, it is `d, c, b, a`.

A set of seen nodes makes each node appear only once, even when several paths lead to it (`a` is reachable through `b` and through `c`). Without it, `a._backward` would run twice and count twice.

### No recursion

The natural version calls itself: `visit(node)` calls `visit(parent)`. Each level of depth then needs a Python stack frame, and Python stops at about 1,000 levels. A long chain of operations is deeper than that; the check builds a chain of 10,000 nodes. So use a list as your own stack instead. Push `(node, False)`. When you pop a node marked False, mark it as seen, push it back as `(node, True)`, and push its unseen parents. When you pop a node marked True, all its parents have been added, so add the node itself.

### Starting values

- The root gets `grad = 1` (the derivative of $L$ with respect to itself), with the shape of its data.
- **Leaves keep adding up**: parameters add new gradients on every call. That is why training calls `zero_grad()` on every parameter before each step.
- **Intermediate nodes start from zero** at the beginning of every `backward` call. So calling `backward` twice gives leaves exactly twice the gradient, instead of a mix of old and new intermediate values.
:::

::: think
Q: Why can't we just call `_backward` in the reverse of the order in which the operations were created?
A: Within one forward pass, creation order is in fact a topological order, so it would work if you recorded it somewhere global. A search from the root needs no global record, works for graphs built in any order, and visits only the nodes that actually affect the root.
:::

::: think
Q: `a` can be reached from the root along two paths. What goes wrong without the set of seen nodes?
A: `a` appears twice in the order, so `a._backward()` runs twice and adds its gradient into its own parents twice. Their gradients double, and in graphs with many such diamonds the work grows very fast.
:::

::: think
Q: You forget to zero the parameters' gradients between two training steps. What does the second step use?
A: The sum of both steps' gradients: step 2 moves the parameters as if the first batch's gradient were still there. Over many steps the gradients keep growing, and the effective step size explodes.
:::

::: hint level=1
Two phases: build the order with your own stack, then walk it backwards and call `_backward`. Between the phases, reset the intermediate gradients and give the root its starting gradient.
:::

::: hint level=2
Use a Python list as the stack and a `set` of `id(node)` values for the seen nodes. Push `(node, False)` for "visit this node" and `(node, True)` for "its parents are done, add it". `np.ones_like(root.data)` is the root's gradient; `reversed(topo)` walks backwards.
:::

::: hint level=3
```pseudo
backward(root):
    topo ← [];  seen ← empty set;  stack ← [(root, false)]
    while stack is not empty:
        node, done ← pop
        if done: add node to topo;  continue
        if node is in seen: continue
        add node to seen;  push (node, true)
        for each parent of node that is not seen yet: push (parent, false)
    for each node in topo that has parents: node.grad ← zeros
    root.grad ← ones like root.data
    for node in topo, from last to first: node._backward()
```
:::

::: attempt
File: `workspace/autograd/tensor.py` — fill in the module-level function `backward(root)`. `Tensor.backward()` already calls it.
Check: `python -m course check a-l03`
:::

::: solution
@include solutions/autograd/tensor.py#backward

- The `(node, expanded)` pairs imitate recursion. A node is added only when it is popped the second time. By then all its parents were pushed later, so they were popped, and added, first.
- `seen` uses `id(node)`, so each node is added once, however many paths lead to it.
- Intermediate gradients are reset at each call; leaves are not. So adding up across calls is explicit, and `zero_grad` controls it.
- `reversed(topo)` starts at the root: every node's gradient is complete before its `_backward` runs.
:::

::: breakit
Write the version that calls itself (`def visit(n): for p in n._prev: visit(p); topo.append(n)`) and run it on the check's 10,000-node chain. Python stops with `RecursionError: maximum recursion depth exceeded` after about 1,000 levels.
:::

::: breakit
Remove the check for seen nodes and run the diamond. `a.grad` comes out larger than 24, because nodes that can be reached twice are processed twice.
:::

::: recap
- `_backward` may run on a node only after all nodes that use it: the reverse of a topological order.
- A depth-first search with your own stack builds that order without hitting Python's recursion limit.
- The root starts with gradient 1; intermediate nodes are reset at each call; leaves add up until `zero_grad`.
:::
