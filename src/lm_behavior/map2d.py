"""Two-dimensional layouts, and how much they cost.

Classical MDS is the default because it preserves the rank of the original
distances better than t-SNE on the small point sets this method produces
(measured: 0.929 against 0.865 on 14 points, 0.975 against 0.821 on 6). t-SNE
is not trying to preserve distances, so that is expected rather than damning --
but a map whose distances cannot be read is a worse map.

Always read `stress` alongside the layout. A plane that retains 55% of the
configuration is a drawing, not a measurement.
"""

from __future__ import annotations

import numpy as np

from ._stats import spearman


def mds(matrix: np.ndarray, dim: int = 2) -> tuple[np.ndarray, float]:
    """Classical (Torgerson) MDS. Returns the layout and the retained fraction."""
    d = np.asarray(matrix, dtype=np.float64)
    n = d.shape[0]
    j = np.eye(n) - np.ones((n, n)) / n
    b = -0.5 * j @ (d**2) @ j
    w, v = np.linalg.eigh((b + b.T) / 2)
    order = np.argsort(w)[::-1]
    w, v = w[order], v[:, order]
    keep = np.clip(w[:dim], 0.0, None)
    coords = v[:, :dim] * np.sqrt(keep)
    positive = np.clip(w, 0.0, None).sum()
    retained = float(keep.sum() / positive) if positive > 0 else 0.0
    return coords, retained


def rank_preservation(matrix: np.ndarray, layout: np.ndarray) -> float:
    """Spearman correlation between original and in-plane pairwise distances."""
    n = matrix.shape[0]
    iu = np.triu_indices(n, 1)
    planar = np.linalg.norm(layout[:, None, :] - layout[None, :, :], axis=-1)
    return spearman(np.asarray(matrix)[iu], planar[iu])


def procrustes(a: np.ndarray, b: np.ndarray) -> float:
    """Disparity between two layouts after centering, scaling and rotation.

    Use it to ask whether a map survived a change -- a different reference set,
    a re-run, a new seed. Swapping the reference *generator* moved a measured
    map 2-3x more than resampling trajectories from the same generator, so this
    number is how you find out whether two maps are comparable at all.
    """
    a = np.asarray(a, dtype=np.float64) - np.mean(a, axis=0)
    b = np.asarray(b, dtype=np.float64) - np.mean(b, axis=0)
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    if na == 0 or nb == 0:
        return float("nan")
    a, b = a / na, b / nb
    u, s, vt = np.linalg.svd(a.T @ b)
    return float(1.0 - (s.sum() ** 2))
