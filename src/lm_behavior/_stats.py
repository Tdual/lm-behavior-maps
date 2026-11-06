"""Pearson and Spearman in numpy, so scipy stays an optional dependency."""

from __future__ import annotations

import numpy as np


def _rank(x: np.ndarray, rtol: float = 1e-12, atol: float = 1e-15) -> np.ndarray:
    """Average ranks, with ties shared.

    Values within a relative tolerance count as tied. Exact equality is the
    wrong test here: these are distances reconstructed through an eigenvector
    decomposition, so a symmetric configuration whose distances are equal in
    exact arithmetic comes back differing in the last bit. Splitting those into
    separate ranks turned a perfect 2-D recovery of a rectangle into a reported
    correlation of 0.956.
    """
    order = np.argsort(x, kind="mergesort")
    ranks = np.empty(len(x), dtype=np.float64)
    ranks[order] = np.arange(len(x), dtype=np.float64)
    sx = x[order]
    i = 0
    while i < len(sx):
        j = i
        while j + 1 < len(sx) and abs(sx[j + 1] - sx[i]) <= atol + rtol * abs(sx[i]):
            j += 1
        if j > i:
            ranks[order[i : j + 1]] = np.mean(ranks[order[i : j + 1]])
        i = j + 1
    return ranks


def pearson(x, y) -> float:
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    if x.std() == 0 or y.std() == 0:
        return float("nan")
    return float(np.corrcoef(x, y)[0, 1])


def spearman(x, y) -> float:
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    return pearson(_rank(x), _rank(y))
