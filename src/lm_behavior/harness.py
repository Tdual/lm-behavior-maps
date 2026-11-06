"""The harness: everything that is not the weights.

History window, system prompt, instruction wording, tool definitions, how much
of the retrieved context is pasted in -- all of it changes the input the model
conditions on, so all of it moves the agent on the map. The chat template
belongs here too: in a six-model measurement, two policies differing only in
whether the template was applied sat at 86% of the median distance between
different models.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, runtime_checkable

from .types import Step, Trajectory


@runtime_checkable
class Harness(Protocol):
    """Builds the prompt body for one decision point.

    The body is plain text. Chat-template wrapping is the scorer's job, since
    the template is model-specific.
    """

    def body(self, trajectory: Trajectory, t: int) -> str: ...


DEFAULT_SYSTEM = (
    "You are an agent. Choose exactly one action from the admissible list to "
    "accomplish the task. Reply with the action only."
)


@dataclass(frozen=True)
class BaseHarness:
    """The paper's BASE harness: task, initial observation, the last `window`
    history blocks, no planning, direct action.

    `window=0` drops the history entirely. `system` is handed to the scorer
    rather than inlined, so that models with a chat template put it in the
    system role and models without one get it as a prefix.
    """

    window: int = 4
    system: str | None = DEFAULT_SYSTEM
    action_cue: str = "Action:"

    def body(self, trajectory: Trajectory, t: int) -> str:
        step: Step = trajectory.steps[t]
        lines = [f"Task: {trajectory.task}", f"Observation: {trajectory.initial_observation}"]
        history = trajectory.history(t)
        for obs, act in (history[-self.window :] if self.window else []):
            lines.append(f"Observation: {obs}")
            lines.append(f"Action: {act}")
        if t > 0 or step.observation != trajectory.initial_observation:
            lines.append(f"Observation: {step.observation}")
        lines.append("Admissible actions:")
        lines += [f"- {a}" for a in step.admissible]
        lines.append(self.action_cue)
        return "\n".join(lines)


@dataclass(frozen=True)
class PrerenderedHarness:
    """Use the prompt body already stored on each step.

    For logs that were captured from a running system, where the prompt is
    whatever that system built. Harness comparison is not available this way --
    every agent sees the same frozen prompt -- so it only answers "are these
    policies the same", not "what does changing the harness do".
    """

    system: str | None = None

    def body(self, trajectory: Trajectory, t: int) -> str:
        context = trajectory.steps[t].context
        if context is None:
            raise ValueError(
                f"step {t} of task {trajectory.task!r} has no pre-rendered context; "
                "use BaseHarness or set Step.context"
            )
        return context


@dataclass(frozen=True)
class FunctionHarness:
    """Wrap any callable `(trajectory, t) -> str`."""

    fn: object
    system: str | None = None

    def body(self, trajectory: Trajectory, t: int) -> str:
        return self.fn(trajectory, t)  # type: ignore[operator]
