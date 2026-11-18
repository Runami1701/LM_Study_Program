"""A tiny reverse-mode autograd engine on NumPy arrays.

Lesson a-l01: Tensor (__init__, __add__, __mul__, zero_grad)  ->  python -m course check a-l01
Lesson a-l02: unbroadcast (+ the ops in ops.py)                ->  python -m course check a-l02
Lesson a-l03: backward                                         ->  python -m course check a-l03
"""

import numpy as np


def unbroadcast(grad: np.ndarray, shape: tuple) -> np.ndarray:
    """Reduce a gradient back to the shape of an operand that was broadcast (a-l02).

    Args:
        grad:  gradient with the (broadcast) shape of the operation's output.
        shape: the operand's own shape before broadcasting.

    Returns:
        An array of exactly `shape`: grad summed over every axis that broadcasting
        added in front, and over every axis where `shape` has length 1 but grad doesn't.
    """
    raise NotImplementedError("unbroadcast")


class Tensor:
    """An array that remembers how it was computed, so it can backpropagate.

    Attributes:
        data:      the NumPy array (float32, or float64 if given float64).
        grad:      array of the same shape and dtype, initially zeros.
        _prev:     tuple of the Tensors this one was computed from (empty for leaves).
        _backward: function of no arguments that adds this tensor's contribution
                   (using self.grad) into the .grad of each parent. Leaves: does nothing.
        _op:       a short label such as "+" (for debugging).
    """

    def __init__(self, data, _prev=(), _op=""):
        """Store data as an ndarray: keep float32/float64 arrays as they are,
        convert anything else (ints, lists, Python floats) to float32."""
        raise NotImplementedError("Tensor")

    @property
    def shape(self) -> tuple:
        return self.data.shape

    def __repr__(self) -> str:
        return f"Tensor(shape={self.data.shape}, dtype={self.data.dtype.name}, op={self._op!r})"

    def _wrap(self, other) -> "Tensor":
        """Turn a number or array into a constant Tensor of this tensor's dtype."""
        return other if isinstance(other, Tensor) else Tensor(np.asarray(other, dtype=self.data.dtype))

    def __add__(self, other) -> "Tensor":
        """self + other. Wrap `other` with self._wrap, create the output Tensor with
        both as parents, and give it a _backward that ACCUMULATES (+=) into both
        parents' grads. (a-l01: same shapes; a-l02: use unbroadcast for broadcasting.)"""
        raise NotImplementedError("Tensor")

    def __mul__(self, other) -> "Tensor":
        """self * other, elementwise, with the same structure as __add__."""
        raise NotImplementedError("Tensor")

    def zero_grad(self) -> None:
        """Reset grad to zeros of data's shape and dtype."""
        raise NotImplementedError("Tensor")

    # ---- given: conveniences built from the methods above ----
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

    def backward(self) -> None:
        """Backpropagate from this scalar tensor (see the module-level backward)."""
        backward(self)


def backward(root: Tensor) -> None:
    """Fill .grad of every tensor that `root` depends on (a-l03).

    1. Collect every tensor reachable from root through _prev, in topological
       order (each tensor after all of its parents). The graph can be thousands
       of nodes deep: don't use recursion.
    2. Reset the grad of every intermediate tensor (one that has parents) to zeros.
       Leaves keep accumulating.
    3. Set root.grad to ones, then call _backward() on every tensor, in reverse
       topological order.
    """
    raise NotImplementedError("backward")
