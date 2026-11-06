"""Amari's alpha-representation, which is where the coordinate comes from.

A distribution can be written as p^((1-alpha)/2), and the choice of alpha is
not a matter of taste -- by Chentsov's theorem the alpha-connections are the
complete list of invariant ones.

    alpha =  1   log p   the paper's coordinate; exponential families are flat,
                         so per-step log-likelihoods add along a trajectory and
                         harness displacements add as vectors
    alpha =  0   sqrt p  an exact isometry onto a sphere, so the distance needs
                         no Taylor expansion and cannot be dominated by one
                         unvisited tail action -- at the cost of flatness
    alpha = -1   p       mixture families are flat

Both ends are provided because the trade-off is real and measurable: on one
36-pair measurement alpha=1 tracked the rank well but overshot the level by
3.5-4.2x, while alpha=0 held the level to 0.87x and never overestimated.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .types import PolicyVectors

EPS = 1e-300


@dataclass
class Coordinates:
    """One agent as a point.

    `z` already carries the 1/sqrt(M) factor, so plain Euclidean distance
    between two of these is the per-step root-mean-square distance.
    """

    name: str
    z: np.ndarray
    alpha: float
    n_steps: int
    mean_length: float
    blocks: list[int] | None = None  # admissible-set size per step, alpha=0 only
    centered: bool = True
    normalized: bool = True

    def __len__(self) -> int:
        return len(self.z)


def embed(
    vectors: PolicyVectors,
    *,
    alpha: float = 0.0,
    normalized: bool = True,
    center: bool = True,
) -> Coordinates:
    """Place one agent.

    `normalized` picks the policy over the admissible set (alpha=1 only). The
    paper's own coordinate is the unnormalised summed token log-probability,
    which has a blind spot worth knowing about: two agents differing only in
    sampling temperature share every token log-probability, so their
    unnormalised distance is exactly zero while their KL is not.

    `center` subtracts the agent's own mean, which removes a per-agent offset
    that no pairwise quantity should depend on. It is ignored at alpha=0, where
    the coordinate lives on a sphere and centering would walk it off.
    """
    m = vectors.n_steps
    if alpha == 1.0:
        v = vectors.recorded_logprob(normalized=normalized)
        if center:
            v = v - v.mean()
        return Coordinates(
            name=vectors.name,
            z=v / np.sqrt(m),
            alpha=1.0,
            n_steps=m,
            mean_length=vectors.mean_length,
            centered=center,
            normalized=normalized,
        )
    if alpha == 0.0:
        blocks = [len(p) for p in vectors.policies]
        z = np.concatenate([np.sqrt(p) for p in vectors.policies]) / np.sqrt(m)
        return Coordinates(
            name=vectors.name,
            z=z,
            alpha=0.0,
            n_steps=m,
            mean_length=vectors.mean_length,
            blocks=blocks,
            centered=False,
            normalized=True,
        )
    if alpha == -1.0:
        blocks = [len(p) for p in vectors.policies]
        z = np.concatenate(list(vectors.policies)) / np.sqrt(m)
        return Coordinates(
            name=vectors.name,
            z=z,
            alpha=-1.0,
            n_steps=m,
            mean_length=vectors.mean_length,
            blocks=blocks,
            centered=False,
            normalized=True,
        )
    raise ValueError(f"alpha must be 1, 0 or -1; got {alpha}")


def embed_trajectory(
    vectors: PolicyVectors, *, normalized: bool = True, center: bool = True
) -> Coordinates:
    """The trajectory-level alpha=1 coordinate: sum within a trajectory first.

    Provided for completeness. It is reliably worse than the step version --
    the theory says its bound is about T-bar times larger, and measurement
    agrees (level overshoot 8.3x against 4.2x on the same data).
    """
    v = vectors.recorded_logprob(normalized=normalized)
    tid = vectors.trajectory_id
    g = np.array([v[tid == t].sum() for t in range(int(tid.max()) + 1)], dtype=np.float64)
    if center:
        g = g - g.mean()
    n = len(g)
    return Coordinates(
        name=vectors.name,
        z=g / np.sqrt(n),
        alpha=1.0,
        n_steps=n,
        mean_length=vectors.mean_length,
        centered=center,
        normalized=normalized,
    )


def embed_all(vectors, **kwargs) -> list[Coordinates]:
    return [embed(v, **kwargs) for v in vectors]
