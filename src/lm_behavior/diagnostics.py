"""Is the method in scope for your agents? Check before trusting the map.

The guarantees cover *close pairs*, and the failure mode when they do not is
not a visible error -- it is a map that looks fine and means nothing. In one
run, six different LLMs on one map gave a rank correlation of 0.39; nine
harness variants of a single model gave 0.97. The difference was not the
locality constant, which was in fact worse in the successful run. It was
whether the pairs spanned a range of true distances at all: in the failed run
every pair sat near the maximum, so there was no order left to recover.

That is checkable in advance, from the policy vectors alone.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field

import numpy as np

from .kl import epsilon, pairwise_exact_kl
from .scoring import shared_admissible
from .types import PolicyVectors


@dataclass
class Diagnosis:
    n_agents: int
    n_pairs: int
    n_steps: int
    kl_min: float
    kl_median: float
    kl_max: float
    spread: float  # max / min; the quantity that decided success in measurement
    saturation: float  # median per-step KL over the maximum entropy of the action set
    epsilon_median: float
    warnings: list[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return not self.warnings

    def __str__(self) -> str:
        head = (
            f"{self.n_agents} agents, {self.n_pairs} pairs, {self.n_steps} reference steps\n"
            f"  exact symmetric KL   min {self.kl_min:.4f}  median {self.kl_median:.4f}  "
            f"max {self.kl_max:.4f}\n"
            f"  spread (max/min)     {self.spread:.1f}x\n"
            f"  saturation           {self.saturation:.2f} of maximum entropy\n"
            f"  epsilon (median)     {self.epsilon_median:.2f}"
        )
        if not self.warnings:
            return head + "\n  no warnings"
        return head + "\n" + "\n".join(f"  WARNING: {w}" for w in self.warnings)


def diagnose(
    vectors: Sequence[PolicyVectors],
    *,
    min_spread: float = 10.0,
    max_saturation: float = 0.5,
) -> Diagnosis:
    """Report whether these agents are in the method's useful range."""
    if len(vectors) < 2:
        raise ValueError("need at least two agents")
    warnings: list[str] = []
    if not shared_admissible(vectors):
        warnings.append(
            "agents do not share the same admissible set at every step; "
            "pairwise quantities are not comparable"
        )

    _, kl = pairwise_exact_kl(vectors)
    iu = np.triu_indices(len(vectors), 1)
    vals = kl[iu]
    positive = vals[vals > 0]
    kl_min = float(positive.min()) if len(positive) else 0.0
    kl_max = float(vals.max())
    spread = (kl_max / kl_min) if kl_min > 0 else float("inf")

    m = vectors[0].n_steps
    per_step = float(np.median(vals)) / vectors[0].mean_length
    max_entropy = float(np.mean([np.log(len(p)) for p in vectors[0].policies]))
    saturation = per_step / (2.0 * max_entropy) if max_entropy > 0 else float("nan")

    eps = [epsilon(vectors[i], vectors[j]) for i, j in zip(*iu, strict=True)]

    if spread < min_spread:
        warnings.append(
            f"true distances span only {spread:.1f}x. The map has little order to "
            f"recover; a measured run at {spread:.1f}x produced a rank correlation of 0.39"
        )
    if saturation > max_saturation:
        warnings.append(
            f"pairs sit at {saturation:.2f} of the maximum possible difference. "
            "These policies are about as different as any two policies could be, "
            "which is outside the close-pair regime the guarantees cover"
        )
    if len(vals) < 10:
        warnings.append(f"only {len(vals)} pairs; correlations will not be meaningful")

    return Diagnosis(
        n_agents=len(vectors),
        n_pairs=int(len(vals)),
        n_steps=m,
        kl_min=kl_min,
        kl_median=float(np.median(vals)),
        kl_max=kl_max,
        spread=spread,
        saturation=saturation,
        epsilon_median=float(np.median(eps)),
        warnings=warnings,
    )
