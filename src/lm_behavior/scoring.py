"""Turn agents and reference trajectories into policy vectors.

One forward pass per (reference step, admissible action). No rollout: nothing
here samples, and nothing here needs an environment to step. That is the whole
cost argument of the method, and it scales linearly in the number of agents.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field

import numpy as np

from .cache import ScoreCache
from .harness import BaseHarness, Harness
from .scorer import Scorer
from .types import PolicyVectors, ReferenceSet


def softmax(raw: np.ndarray, temperature: float = 1.0) -> np.ndarray:
    """The admissible-set policy: a softmax over summed action log-probabilities.

    Temperature belongs to the harness, not to the weights: two temperatures of
    one model share every token log-probability and are still two agents.
    """
    v = np.asarray(raw, dtype=np.float64) / temperature
    p = np.exp(v - v.max())
    return p / p.sum()


@dataclass
class Agent:
    """An LLM plus a harness. Either half alone is not an agent."""

    name: str
    scorer: Scorer
    harness: Harness = field(default_factory=BaseHarness)
    temperature: float = 1.0

    @property
    def system(self) -> str | None:
        return getattr(self.harness, "system", None)


def score_references(
    agent: Agent,
    references: ReferenceSet,
    *,
    cache: ScoreCache | None = None,
    progress: bool = False,
) -> PolicyVectors:
    """Score every admissible action at every reference step."""
    bodies: list[str] = []
    actions: list[list[str]] = []
    action_index: list[int] = []
    trajectory_id: list[int] = []

    for r, t, traj, step in references.flat():
        bodies.append(agent.harness.body(traj, t))
        actions.append(list(step.admissible))
        action_index.append(step.action_index)
        trajectory_id.append(r)

    fp = agent.scorer.fingerprint
    system = agent.system

    if hasattr(agent.scorer, "policy"):
        return _score_by_policy(
            agent, references, bodies, actions, action_index, trajectory_id, cache, progress
        )

    wanted: list[tuple[str, str]] = []
    slots: list[tuple[int, int]] = []
    scores: list[np.ndarray] = [np.empty(len(a), dtype=np.float64) for a in actions]

    for k, (body, acts) in enumerate(zip(bodies, actions, strict=True)):
        for i, act in enumerate(acts):
            hit = cache.get(fp, system, body, act) if cache else None
            if hit is None:
                wanted.append((body, act))
                slots.append((k, i))
            else:
                scores[k][i] = hit

    if wanted:
        if progress:
            print(f"[{agent.name}] scoring {len(wanted)} of "
                  f"{sum(len(a) for a in actions)} (body, action) pairs")
        got = agent.scorer.loglikelihood(wanted, system=system)
        if len(got) != len(wanted):
            raise RuntimeError(
                f"scorer returned {len(got)} values for {len(wanted)} requests"
            )
        for (k, i), v in zip(slots, got, strict=True):
            scores[k][i] = v
        if cache:
            cache.put_many(fp, system, ((b, a, v) for (b, a), v in zip(wanted, got, strict=True)))

    policies = [softmax(s, agent.temperature) for s in scores]
    return PolicyVectors(
        name=agent.name,
        policies=policies,
        raw=scores,
        action_index=action_index,
        trajectory_id=np.asarray(trajectory_id, dtype=np.int64),
        mean_length=references.mean_length,
        meta={
            "scorer": fp,
            "temperature": agent.temperature,
            "harness": type(agent.harness).__name__,
            "system": system,
            "generator": references.generator,
        },
    )


def _score_by_policy(
    agent, references, bodies, actions, action_index, trajectory_id, cache, progress
) -> PolicyVectors:
    """Backends that hand back a whole distribution: one call per step.

    The cache still stores one value per (body, action), so a policy-based and
    a teacher-forced run of the same agent do not share entries -- they are
    different quantities and the fingerprints differ accordingly.
    """
    fp, system = agent.scorer.fingerprint, agent.system
    policies: list[np.ndarray] = []
    fetched = 0

    for body, acts in zip(bodies, actions, strict=True):
        cached = [cache.get(fp, system, body, a) if cache else None for a in acts]
        if all(c is not None for c in cached):
            p = np.asarray(cached, dtype=np.float64)
        else:
            fetched += 1
            p = np.asarray(agent.scorer.policy(body, acts, system=system), dtype=np.float64)
            if p.shape != (len(acts),):
                raise RuntimeError(f"policy() returned shape {p.shape} for {len(acts)} actions")
            if np.any(p < 0):
                raise RuntimeError("policy() returned a negative probability")
            if cache:
                cache.put_many(
                    fp, system, [(body, a, v) for a, v in zip(acts, p, strict=True)]
                )
        total = float(p.sum())
        if total <= 0:
            raise RuntimeError("policy() returned an all-zero distribution")
        p = p / total
        if agent.temperature != 1.0:
            p = p ** (1.0 / agent.temperature)
            p = p / p.sum()
        policies.append(p)

    if progress:
        print(f"[{agent.name}] fetched {fetched} of {len(bodies)} steps, rest from cache")

    return PolicyVectors(
        name=agent.name,
        policies=policies,
        raw=[np.log(p + 1e-300) for p in policies],
        action_index=action_index,
        trajectory_id=np.asarray(trajectory_id, dtype=np.int64),
        mean_length=references.mean_length,
        meta={
            "scorer": fp,
            "temperature": agent.temperature,
            "harness": type(agent.harness).__name__,
            "system": system,
            "generator": references.generator,
            "unnormalised_coordinate": False,
        },
    )


def score_all(
    agents: Iterable[Agent],
    references: ReferenceSet,
    *,
    cache: ScoreCache | None = None,
    progress: bool = False,
) -> list[PolicyVectors]:
    """Score several agents against one reference set, in order."""
    return [score_references(a, references, cache=cache, progress=progress) for a in agents]


def shared_admissible(vectors: Sequence[PolicyVectors]) -> bool:
    """Do these agents share the same admissible set at every step?

    Required before any pairwise quantity is computed: two policies over
    different supports are not comparable.
    """
    sizes = [[len(p) for p in v.policies] for v in vectors]
    return all(s == sizes[0] for s in sizes[1:])
