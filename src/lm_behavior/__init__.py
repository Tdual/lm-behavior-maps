"""lm-behavior-maps: locate LLM agents by what they would do.

Each agent teacher-forces the actions of a shared set of reference
trajectories. The per-step log-likelihoods become its coordinates, and the
distance between two agents tracks the KL divergence between their policies --
without running either of them.

    import lm_behavior as lb

    refs    = lb.load_references("refs.json")
    scorer  = lb.HFScorer("Qwen/Qwen2.5-0.5B-Instruct")
    agents  = [lb.Agent("w4", scorer, lb.BaseHarness(window=4)),
               lb.Agent("w1", scorer, lb.BaseHarness(window=1))]

    vectors = lb.score_all(agents, refs, cache=lb.ScoreCache())
    print(lb.diagnose(vectors))                 # is this in scope at all?
    names, d = lb.pairwise(lb.embed_all(vectors, alpha=0))

Read the distance as a rank, not as a number of nats. See README for what the
method does and does not deliver.
"""

from .cache import ScoreCache
from .compose import Composition, compose_check
from .diagnostics import Diagnosis, diagnose
from .distance import (
    bhattacharyya,
    distance,
    pairwise,
    symmetric_kl_estimate,
    to_pairs,
)
from .embed import Coordinates, embed, embed_all, embed_trajectory
from .harness import BaseHarness, FunctionHarness, Harness, PrerenderedHarness
from .kl import epsilon, exact_symmetric_kl, pairwise_exact_kl
from .map2d import mds, procrustes, rank_preservation
from .references import load as load_references
from .references import save as save_references
from .scorer import Scorer
from .scorers import FunctionScorer
from .scoring import Agent, score_all, score_references, shared_admissible, softmax
from .tangent import fisher_rao_log, step_angles
from .types import PolicyVectors, ReferenceSet, Step, Trajectory

__version__ = "0.1.0"

__all__ = [
    "Agent",
    "BaseHarness",
    "Composition",
    "Coordinates",
    "Diagnosis",
    "FunctionHarness",
    "FunctionScorer",
    "Harness",
    "PolicyVectors",
    "PrerenderedHarness",
    "ReferenceSet",
    "ScoreCache",
    "Scorer",
    "Step",
    "Trajectory",
    "__version__",
    "bhattacharyya",
    "compose_check",
    "diagnose",
    "distance",
    "embed",
    "embed_all",
    "embed_trajectory",
    "epsilon",
    "exact_symmetric_kl",
    "fisher_rao_log",
    "load_references",
    "mds",
    "pairwise",
    "pairwise_exact_kl",
    "procrustes",
    "rank_preservation",
    "save_references",
    "score_all",
    "score_references",
    "shared_admissible",
    "softmax",
    "step_angles",
    "symmetric_kl_estimate",
    "to_pairs",
]


def __getattr__(name):
    if name in ("HFScorer", "LMEvalScorer"):
        from . import scorers

        return getattr(scorers, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
