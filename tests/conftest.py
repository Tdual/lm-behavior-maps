import hashlib

import numpy as np
import pytest

import lm_behavior as lb


def toy_logit(body: str, action: str, seed: int) -> float:
    """Deterministic pseudo-logit: a fixed function of (seed, body, action)."""
    h = hashlib.blake2b(f"{seed}|{body}|{action}".encode(), digest_size=8).digest()
    return (int.from_bytes(h, "big") / 2**64) * 6.0 - 3.0


def toy_scorer(seed: int, name: str | None = None) -> lb.FunctionScorer:
    return lb.FunctionScorer(
        fn=lambda body, action, system, s=seed: toy_logit(body, action, s),
        name=name or f"toy{seed}",
    )


@pytest.fixture
def refs() -> lb.ReferenceSet:
    rng = np.random.default_rng(0)
    trajectories = []
    for r in range(4):
        steps = []
        for t in range(5):
            admissible = [f"act{r}_{t}_{k}" for k in range(4)]
            steps.append(
                lb.Step(
                    observation=f"obs {r}.{t}",
                    admissible=admissible,
                    action=admissible[int(rng.integers(len(admissible)))],
                )
            )
        trajectories.append(
            lb.Trajectory(task=f"task {r}", initial_observation=f"start {r}", steps=steps)
        )
    return lb.ReferenceSet(trajectories=trajectories, generator="test-fixture")


@pytest.fixture
def vectors(refs):
    agents = [
        lb.Agent("a", toy_scorer(1), lb.BaseHarness(window=4)),
        lb.Agent("b", toy_scorer(2), lb.BaseHarness(window=4)),
        lb.Agent("c", toy_scorer(3), lb.BaseHarness(window=4)),
    ]
    return lb.score_all(agents, refs)
