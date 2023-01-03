"""Binary node."""
import functools
import numpy as np


@functools.total_ordering
class Bnode:
    """Binary node.

    In binary segmentation, each segment [start, end) is a binary node.
    """

    def __init__(self, start, end, val, left=None, right=None, parent=None):
        self.start = start
        self.end = end
        self.val = val
        self.left = left
        self.right = right
        self.parent = parent

    @property
    def gain(self):
        """Return the cost decrease when splitting this node."""
        if self.left is None or self.right is None:
            return 0
        elif np.isinf(self.val) and self.val < 0:
            return 0
        return self.val - (self.left.val + self.right.val)

    def __lt__(self, other):
        return self.start < other.start

    def __eq__(self, other):
        return (
            isinstance(other, self.__class__)
            and self.start == other.start
            and self.end == other.end
        )

    def __hash__(self):
        return hash((self.__class__, self.start, self.end))
