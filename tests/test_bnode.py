import numpy as np
from ruptures.utils import Bnode


def test_bnode():
    left = Bnode(start=100, end=120, val=1)
    right = Bnode(start=120, end=200, val=1)

    # bad merging, no right leaf
    merged_node = Bnode(start=left.start, end=right.end, left=left, right=None, val=3)
    assert merged_node.gain == 0

    # bad merging, no left leaf
    merged_node = Bnode(start=left.start, end=right.end, left=None, right=right, val=3)
    assert merged_node.gain == 0

    # bad merging, negative infinit val
    merged_node = Bnode(
        start=left.start, end=right.end, left=left, right=right, val=-np.inf
    )
    assert merged_node.gain == 0

    # normal merging
    merged_node = Bnode(
        start=left.start,
        end=right.end,
        left=left,
        right=right,
        val=left.val + right.val + 1,
    )
    assert merged_node.gain == merged_node.val - (left.val + right.val)
