"""Do two harness changes add up?

The paper's Appendix C asks whether the displacement caused by applying two
harness changes together equals the sum of the displacements each causes alone.
If it does, a combination can be predicted without measuring it.

Measured answer, on both alpha=1 and alpha=0: the *direction* adds (cosine
0.93-0.99), the *magnitude* does not (the sum overshoots by about 1.18x). And
the gap is structural, not noise -- two changes that remove the same thing (a
shorter history window and a dropped system prompt both cut input information)
saturate, so their effects overlap instead of stacking.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

from .embed import embed
from .tangent import fisher_rao_log
from .types import PolicyVectors


@dataclass
class Composition:
    """How well `parts` predicted `combined`, relative to `base`."""

    cosine: float
    norm_ratio: float  # ||measured|| / ||predicted||; below 1 means the sum overshoots
    relative_error: float

    def as_dict(self) -> dict[str, float]:
        return {
            "cosine": self.cosine,
            "norm_ratio": self.norm_ratio,
            "relative_error": self.relative_error,
        }


def _displacements(base, combined, parts, alpha, normalized):
    if alpha == 0.0:
        measured = fisher_rao_log(combined, base)
        predicted = sum(fisher_rao_log(p, base) for p in parts)
        return measured, predicted
    kw = {"alpha": 1.0, "center": True, "normalized": normalized}
    zb = embed(base, **kw).z
    measured = embed(combined, **kw).z - zb
    predicted = sum(embed(p, **kw).z - zb for p in parts)
    return measured, predicted


def compose_check(
    base: PolicyVectors,
    combined: PolicyVectors,
    parts: Sequence[PolicyVectors],
    *,
    alpha: float = 1.0,
    normalized: bool = True,
) -> Composition:
    """Compare the measured displacement with the sum of the single-change ones.

    At alpha=0 the differences are taken through the Fisher-Rao log map at
    `base`, because the raw differences are chords and chords do not add.

    `normalized` selects the policy coordinate over the paper's unnormalised
    one at alpha=1. Only the unnormalised coordinate is linear in the scorer's
    own output, so a change that is additive in the logits is additive here
    exactly; the normalised coordinate puts a softmax in the way.
    """
    measured, predicted = _displacements(base, combined, parts, alpha, normalized)
    nm, np_ = float(np.linalg.norm(measured)), float(np.linalg.norm(predicted))
    if nm == 0.0 or np_ == 0.0:
        raise ValueError("a displacement is exactly zero; nothing to compare")
    return Composition(
        cosine=float(np.dot(measured, predicted) / (nm * np_)),
        norm_ratio=nm / np_,
        relative_error=float(np.linalg.norm(predicted - measured) / nm),
    )
