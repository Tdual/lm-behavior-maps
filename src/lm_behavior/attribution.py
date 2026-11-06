"""Which component moved, and at which steps.

A single distance says a configuration changed. For a service with a router, a
specialist and a synthesiser, the useful question is *which of them* changed --
and whether a component nobody edited moved anyway, which is the interesting
failure.

Both quantities decompose exactly over steps, so the parts add up to the whole
with no residual: the symmetric KL is a sum over steps by construction, and the
alpha=0 squared distance is a mean of per-step squared chords. Shares are
therefore real shares, not an approximate split.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

from .kl import per_step_kl
from .types import PolicyVectors


@dataclass
class RoleContribution:
    role: str
    n_steps: int
    total: float  # this role's contribution, in the same units as the whole
    share: float  # fraction of the total
    per_step: float  # mean over this role's steps, for comparing unequal roles


def _per_step(a: PolicyVectors, b: PolicyVectors, metric: str) -> np.ndarray:
    if a.n_steps != b.n_steps:
        raise ValueError("policy vectors come from different reference sets")
    out = np.empty(a.n_steps, dtype=np.float64)
    for k, (p, q) in enumerate(zip(a.policies, b.policies, strict=True)):
        if len(p) != len(q):
            raise ValueError(f"admissible sets differ at step {k}")
        if metric == "kl":
            out[k] = per_step_kl(p, q) + per_step_kl(q, p)
        elif metric == "hellinger":
            out[k] = float(((np.sqrt(p) - np.sqrt(q)) ** 2).sum())
        else:
            raise ValueError(f"unknown metric {metric!r}; use 'kl' or 'hellinger'")
    return out


def _roles(a: PolicyVectors, b: PolicyVectors) -> list[str]:
    source = a.roles or b.roles
    if not source:
        return ["(unset)"] * a.n_steps
    if b.roles and list(a.roles) != list(b.roles):
        raise ValueError("the two runs label their steps with different roles")
    return [r or "(unset)" for r in source]


def by_role(
    a: PolicyVectors, b: PolicyVectors, *, metric: str = "kl"
) -> list[RoleContribution]:
    """Split the distance between two configurations across components.

    Ordered by contribution, largest first. `per_step` is what to read when the
    roles own very different numbers of steps: a synthesiser called once per
    session and a specialist called six times are not comparable on totals.
    """
    values = _per_step(a, b, metric)
    roles = _roles(a, b)
    scale = a.mean_length / a.n_steps  # KL per trajectory, as the pairwise API reports it
    total = float(values.sum()) * scale

    out: list[RoleContribution] = []
    for role in dict.fromkeys(roles):
        mask = np.array([r == role for r in roles])
        part = float(values[mask].sum()) * scale
        out.append(
            RoleContribution(
                role=role,
                n_steps=int(mask.sum()),
                total=part,
                share=(part / total) if total > 0 else 0.0,
                per_step=float(values[mask].mean()),
            )
        )
    return sorted(out, key=lambda r: r.total, reverse=True)


def top_steps(
    a: PolicyVectors, b: PolicyVectors, *, n: int = 10, metric: str = "kl"
) -> list[tuple[int, str, float]]:
    """The individual reference steps that moved most: (index, role, value).

    Where to look first when a component moved and the reason is not obvious.
    """
    values = _per_step(a, b, metric)
    roles = _roles(a, b)
    order = np.argsort(values)[::-1][:n]
    return [(int(k), roles[k], float(values[k])) for k in order]


def format_by_role(contributions: Sequence[RoleContribution], metric: str = "kl") -> str:
    width = max((len(c.role) for c in contributions), default=4) + 2
    head = f"{'role':<{width}}{'steps':>7}{'total':>11}{'share':>8}{'per step':>11}"
    lines = [head, "-" * len(head)]
    for c in contributions:
        lines.append(
            f"{c.role:<{width}}{c.n_steps:>7}{c.total:>11.4f}"
            f"{c.share:>7.1%}{c.per_step:>11.4f}"
        )
    lines.append(f"({metric}, symmetric, scaled to one trajectory)")
    return "\n".join(lines)
