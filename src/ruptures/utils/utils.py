"""Miscellaneous functions for ruptures."""

from typing import Iterable, TypeVar, Iterator
from itertools import tee
from math import ceil

T = TypeVar("T")


def pairwise(iterable: Iterable[T]) -> Iterator[tuple[T, T]]:
    """S -> (s0,s1), (s1,s2), (s2, s3), ..."""
    a, b = tee(iterable)
    next(b, None)
    return zip(a, b)


def unzip(seq: Iterable[tuple[T, ...]]) -> Iterator[tuple[T, ...]]:
    """Reverse zip."""
    return zip(*seq)


def sanity_check(n_samples: int, n_bkps: int, jump: int, min_size: int) -> bool:
    """Check if a partition if possible given some segmentation parameters.

    Args:
        n_samples (int): number of point in the signal
        n_bkps (int): number of breakpoints
        jump (int): the start index of each regime can only be a multiple of
            "jump" (and the end index = -1 modulo "jump").
        min_size (int): minimum size of a segment.

    Returns:
        bool: True if there exists a potential configuration of
            breakpoints for the given parameters. False if it does not.
    """
    n_adm_bkps = n_samples // jump  # number of admissible breakpoints

    # Are there enough points for the given number of regimes?
    if n_bkps > n_adm_bkps:
        return False
    if n_bkps * ceil(min_size / jump) * jump + min_size > n_samples:
        return False
    return True
