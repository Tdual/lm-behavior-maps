"""Distances between placed agents, and what they estimate.

The squared coordinate distance is meant to track the symmetric trajectory KL.
It tracks the *order* well. It does not deliver the level: treat a distance as
a rank, not as a calibrated number of nats, unless you are at alpha=0 and have
checked the ratio on your own data.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from .embed import Coordinates


def _blocks(c: Coordinates) -> list[slice]:
    out, start = [], 0
    for n in c.blocks or []:
        out.append(slice(start, start + n))
        start += n
    return out


def bhattacharyya(a: Coordinates, b: Coordinates) -> np.ndarray:
    """Per-step Bhattacharyya coefficient, from two alpha=0 coordinates."""
    if a.alpha != 0.0 or b.alpha != 0.0:
        raise ValueError("bhattacharyya needs alpha=0 coordinates")
    m = a.n_steps
    bc = np.array(
        [float(np.dot(a.z[s], b.z[s])) for s in _blocks(a)], dtype=np.float64
    ) * m
    return np.clip(bc, 0.0, 1.0)


def distance(a: Coordinates, b: Coordinates, metric: str | None = None) -> float:
    """Per-step root-mean-square distance between two agents.

    metric: "euclidean" (the default at alpha=1 and alpha=-1), "hellinger" (the
    chord, default at alpha=0) or "fisher_rao" (the geodesic arc). Chord and
    arc differ by 2-4% on measured data; the arc is the honest length on the
    sphere, the chord is what a flat tool would compute.
    """
    if a.alpha != b.alpha:
        raise ValueError("cannot compare coordinates built with different alpha")
    if len(a.z) != len(b.z):
        raise ValueError("coordinates have different lengths; same reference set?")
    metric = metric or ("hellinger" if a.alpha == 0.0 else "euclidean")
    if metric in ("euclidean", "hellinger"):
        return float(np.linalg.norm(a.z - b.z))
    if metric == "fisher_rao":
        theta = np.arccos(bhattacharyya(a, b))
        return float(np.sqrt(np.mean(theta**2)))
    raise ValueError(f"unknown metric {metric!r}")


def symmetric_kl_estimate(a: Coordinates, b: Coordinates, metric: str | None = None) -> float:
    """Estimate the symmetric trajectory KL from the coordinate distance.

    alpha=1:  T_bar * D^2        (the paper's (T_bar/2) D^2 per direction)
    alpha=0:  4 * T_bar * D^2    (KL ~ 2||sqrt p - sqrt q||^2 per direction)
    """
    d = distance(a, b, metric)
    factor = 4.0 if a.alpha == 0.0 else 1.0
    return factor * a.mean_length * d * d


def pairwise(
    coords: Sequence[Coordinates], metric: str | None = None, estimate_kl: bool = False
) -> tuple[list[str], np.ndarray]:
    """Full symmetric matrix over a list of placed agents."""
    n = len(coords)
    out = np.zeros((n, n), dtype=np.float64)
    fn = symmetric_kl_estimate if estimate_kl else distance
    for i in range(n):
        for j in range(i + 1, n):
            out[i, j] = out[j, i] = fn(coords[i], coords[j], metric)
    return [c.name for c in coords], out


def to_pairs(names: Sequence[str], matrix: np.ndarray) -> dict[tuple[str, str], float]:
    """The upper triangle as a {(name_i, name_j): value} mapping."""
    return {
        (names[i], names[j]): float(matrix[i, j])
        for i in range(len(names))
        for j in range(i + 1, len(names))
    }
