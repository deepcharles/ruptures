"""Binary node."""
import numpy as np


class Bnode:

    """Binary node.

    In binary segmentation, each segment [start, end) is a binary node.

    """

    def __init__(self, start, end, val, left=None, right=None):
        self.start = start
        self.end = end
        self.val = val
        self.left = left
        self.right = right

    @property
    def loss(self):
        """Return the cost loss when splitting this node."""
        if self.left is None or self.right is None:
            return np.inf
        return (self.left.val + self.right.val) / self.val
