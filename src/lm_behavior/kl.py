"""Exact KL under the reference distribution -- no embedding involved.

The forward pass already produced a score for every admissible action. Keeping
the whole policy vector instead of one entry gives the exact per-step KL for
every pair, at the same number of forward passes, in closed form.

That makes this the ceiling for the method: every reference-set quantity is an
approximation of *this*, so how well this predicts the on-policy KL bounds what
any embedding can achieve. On one measured set that ceiling was Spearman 0.931,
and an embedding scoring above it was not tracking KL better -- it was
correlating with the reference mismatch.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from .types import PolicyVectors

EPS = 1e-300


def per_step_kl(p: np.ndarray, q: np.ndarray) -> float:
    """KL(p || q) over one admissible set."""
    return float((p * (np.log(p + EPS) - np.log(q + EPS))).sum())


def exact_symmetric_kl(a: PolicyVectors, b: PolicyVectors) -> float:
    """Symmetric trajectory KL under the reference distribution."""
    if a.n_steps != b.n_steps:
        raise ValueError("policy vectors come from different reference sets")
    total = 0.0
    for p, q in zip(a.policies, b.policies, strict=True):
        if len(p) != len(q):
            raise ValueError("admissible sets differ between agents at the same step")
        total += per_step_kl(p, q) + per_step_kl(q, p)
    return a.mean_length * total / a.n_steps


def pairwise_exact_kl(vectors: Sequence[PolicyVectors]) -> tuple[list[str], np.ndarray]:
    n = len(vectors)
    out = np.zeros((n, n), dtype=np.float64)
    for i in range(n):
        for j in range(i + 1, n):
            out[i, j] = out[j, i] = exact_symmetric_kl(vectors[i], vectors[j])
    return [v.name for v in vectors], out


def epsilon(a: PolicyVectors, b: PolicyVectors) -> float:
    """The locality quantity of the paper's Assumption 1: the largest per-step
    log-ratio over the admissible set.

    Reported because the theory is stated in it, not because it is a usable
    gate. It is a maximum over *all* admissible actions, so one action the
    policy never takes can dominate it: measured medians above 10 coexist with
    rank correlations of 0.96. The bound is sufficient, not necessary.
    """
    worst = 0.0
    for p, q in zip(a.policies, b.policies, strict=True):
        worst = max(worst, float(np.abs(np.log(p + EPS) - np.log(q + EPS)).max()))
    return worst
