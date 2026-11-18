"""A tiny reverse-mode autograd engine on NumPy arrays (reference solution)."""

import numpy as np


def unbroadcast(grad: np.ndarray, shape: tuple) -> np.ndarray:
    """Sum `grad` over the axes that broadcasting added or stretched, so the
    result has `shape` (the shape of the operand before broadcasting)."""
    # 1) axes that broadcasting prepended: sum them away
    while grad.ndim > len(shape):
        grad = grad.sum(axis=0)
    # 2) axes where the operand had length 1 but the result didn't: sum, keep the axis
    for axis, size in enumerate(shape):
        if size == 1 and grad.shape[axis] != 1:
            grad = grad.sum(axis=axis, keepdims=True)
    return grad


class Tensor:
    """An array that remembers how it was computed, so it can backpropagate."""

    def __init__(self, data, _prev=(), _op=""):
        arr = np.asarray(data)
        if arr.dtype not in (np.float32, np.float64):
            arr = arr.astype(np.float32)          # ints, Python floats -> float32
        self.data = arr
        self.grad = np.zeros(arr.shape, dtype=arr.dtype)   # np.zeros, not zeros_like: see a-l05
        self._prev = tuple(_prev)                 # the Tensors this one was computed from
        self._backward = lambda: None             # pushes self.grad into the parents' grads
        self._op = _op

    @property
    def shape(self) -> tuple:
        return self.data.shape

    def __repr__(self) -> str:
        return f"Tensor(shape={self.data.shape}, dtype={self.data.dtype.name}, op={self._op!r})"

    def _wrap(self, other) -> "Tensor":
        return other if isinstance(other, Tensor) else Tensor(np.asarray(other, dtype=self.data.dtype))

    def __add__(self, other) -> "Tensor":
        other = self._wrap(other)
        out = Tensor(self.data + other.data, (self, other), "+")

        def _backward():
            self.grad += unbroadcast(out.grad, self.data.shape)    # d(a+b)/da = 1
            other.grad += unbroadcast(out.grad, other.data.shape)  # d(a+b)/db = 1

        out._backward = _backward
        return out

    def __mul__(self, other) -> "Tensor":
        other = self._wrap(other)
        out = Tensor(self.data * other.data, (self, other), "*")

        def _backward():
            self.grad += unbroadcast(out.grad * other.data, self.data.shape)   # d(ab)/da = b
            other.grad += unbroadcast(out.grad * self.data, other.data.shape)  # d(ab)/db = a

        out._backward = _backward
        return out

    # Conveniences built from + and * (given, not part of any lesson).
    def __radd__(self, other):
        return self + other

    def __rmul__(self, other):
        return self * other

    def __neg__(self):
        return self * -1.0

    def __sub__(self, other):
        return self + (-self._wrap(other))

    def __rsub__(self, other):
        return self._wrap(other) + (-self)

    def __matmul__(self, other):
        from .ops import matmul
        return matmul(self, self._wrap(other))

    def __getitem__(self, key):
        from .ops import getitem
        return getitem(self, key)

    def zero_grad(self) -> None:
        """Reset this tensor's gradient to zeros (same shape and dtype as data)."""
        self.grad = np.zeros_like(self.data)

    def backward(self) -> None:
        """Backpropagate from this scalar tensor (see the module-level backward)."""
        backward(self)


def backward(root: Tensor) -> None:
    """Fill .grad of every tensor that `root` depends on.

    Leaf tensors (no parents: parameters, inputs) ACCUMULATE into .grad, so call
    zero_grad on them between training steps. Intermediate tensors are reset first.
    """
    topo, seen = [], set()
    stack = [(root, False)]
    while stack:                                  # iterative DFS: deep graphs can't overflow the stack
        node, expanded = stack.pop()
        if expanded:
            topo.append(node)                     # all parents already placed: post-order
            continue
        if id(node) in seen:
            continue
        seen.add(id(node))
        stack.append((node, True))
        for parent in node._prev:
            if id(parent) not in seen:
                stack.append((parent, False))
    for node in topo:
        if node._prev:                            # intermediate: start from zero every time
            node.grad = np.zeros(node.data.shape, dtype=node.data.dtype)
    root.grad = np.ones_like(root.data)           # dL/dL = 1
    for node in reversed(topo):                   # each node runs after every node that uses it
        node._backward()
