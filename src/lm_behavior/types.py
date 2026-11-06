"""Data the method operates on.

A *reference trajectory* is raw material, not a rendered prompt. The prompt is
built from it by a harness, and the harness is part of the agent -- two agents
that share weights but differ in history window, system prompt or chat template
are two different points on the map. Storing rendered prompts would freeze one
harness into the data and make harness comparison impossible.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass
class Step:
    """One decision point of a reference trajectory.

    `admissible` is the finite, known set of actions available here. It is what
    makes the per-step KL a closed-form sum rather than a density estimate, and
    it is required: the method has no defined policy without it.

    `action` is the action actually recorded at this step. It must be a member
    of `admissible`.

    `role` names the component that made this decision -- "router",
    "specialist", "synthesiser". A single-agent system leaves it unset. It is
    what lets one map hold a service whose steps are decided by different
    models under different prompts, and what lets a change be attributed to the
    component that caused it.
    """

    observation: str
    admissible: list[str]
    action: str
    context: str | None = None  # pre-rendered prompt body, for PrerenderedHarness
    role: str | None = None  # which component of a multi-agent system decided here
    meta: dict = field(default_factory=dict)  # raw messages and tools, when imported

    def __post_init__(self) -> None:
        if not self.admissible:
            raise ValueError("a step needs a non-empty admissible set")
        if self.action not in self.admissible:
            raise ValueError(f"recorded action {self.action!r} is not in the admissible set")

    @property
    def action_index(self) -> int:
        return self.admissible.index(self.action)


@dataclass(frozen=True)
class Trajectory:
    """A sequence of decision points sharing one task."""

    task: str
    initial_observation: str
    steps: list[Step]

    def history(self, t: int) -> list[tuple[str, str]]:
        """The (observation, action) pairs before step `t`, oldest first."""
        return [(s.observation, s.action) for s in self.steps[:t]]


@dataclass(frozen=True)
class ReferenceSet:
    """The shared reference trajectories every agent is scored against.

    The generator of these trajectories is part of the measurement, not a
    neutral backdrop: swapping it moves the resulting map several times more
    than resampling trajectories from the same generator. Record what produced
    them in `generator` so two maps can be checked for comparability.
    """

    trajectories: list[Trajectory]
    generator: str | None = None

    @property
    def n_steps(self) -> int:
        return sum(len(t.steps) for t in self.trajectories)

    @property
    def n_trajectories(self) -> int:
        return len(self.trajectories)

    @property
    def mean_length(self) -> float:
        """T-bar: the mean number of steps per trajectory."""
        return self.n_steps / self.n_trajectories

    def flat(self):
        """Yield (trajectory_index, step_index, trajectory, step) in order."""
        for r, traj in enumerate(self.trajectories):
            for t, step in enumerate(traj.steps):
                yield r, t, traj, step


@dataclass
class PolicyVectors:
    """One agent's policy over the admissible set at every reference step.

    `policies[k]` sums to one over the admissible set of reference step k.
    `raw[k]` holds the unnormalised summed token log-probabilities the scorer
    returned, which is the paper's own coordinate and is kept so both
    conventions can be compared rather than assumed.
    """

    name: str
    policies: list[np.ndarray]
    raw: list[np.ndarray]
    action_index: list[int]
    trajectory_id: np.ndarray
    mean_length: float
    roles: list[str | None] = field(default_factory=list)
    meta: dict = field(default_factory=dict)

    @property
    def n_steps(self) -> int:
        return len(self.policies)

    def recorded_logprob(self, normalized: bool = True) -> np.ndarray:
        """The per-step coordinate before centering: log pi(a_recorded | h)."""
        src = self.policies if normalized else None
        if src is None:
            return np.array(
                [float(self.raw[k][i]) for k, i in enumerate(self.action_index)],
                dtype=np.float64,
            )
        return np.array(
            [float(np.log(src[k][i] + 1e-300)) for k, i in enumerate(self.action_index)],
            dtype=np.float64,
        )
