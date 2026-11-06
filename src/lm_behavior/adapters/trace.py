"""The contract every framework adapter produces.

A trace step is one decision an LLM made: the conversation it saw, the set of
options it had, and which one it took. That is all the method needs, and it is
the piece every agent framework has in common, so the adapters are thin and the
contract is the thing to write against if yours is not supported.

The admissible set comes from the trace, which decides what a comparison can
mean. Two configurations are compared over the options that were on the table
when the trace was recorded. Change a prompt, a model, a temperature, or the
wording of a skill, and both configurations face the same options -- the
comparison is exact. *Add* a tool and the new configuration has an option the
trace does not contain: what gets measured is how its preference over the old
options shifted, not how often it reaches for the new one. That is a real
measurement and a real limitation, and `extra_actions` is there for when you
would rather put the new option in front of both.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field

from ..types import ReferenceSet, Step, Trajectory

FINISH = "__finish__"


@dataclass
class TraceStep:
    """One recorded decision."""

    session: str  # groups steps into a trajectory; a thread, a run, a crew kickoff
    role: str  # which component decided: a node name, an agent role
    messages: list[dict]  # the conversation, OpenAI shape: {"role", "content"}
    admissible: list[str]  # the options, usually tool names
    action: str  # the one taken
    meta: dict = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.action not in self.admissible:
            raise ValueError(
                f"recorded action {self.action!r} is not among the options "
                f"{self.admissible!r} at role {self.role!r}"
            )


def render(messages: Sequence[dict], *, skip_system: bool) -> str:
    """Flatten a conversation to the plain text a scorer conditions on."""
    lines = []
    for m in messages:
        role = str(m.get("role", "user"))
        if skip_system and role == "system":
            continue
        content = m.get("content")
        if isinstance(content, list):  # multimodal parts
            content = "".join(
                part.get("text", "") for part in content if isinstance(part, dict)
            )
        lines.append(f"{role}: {content}")
    return "\n".join(lines)


def build_references(
    steps: Sequence[TraceStep],
    *,
    generator: str | None = None,
    extra_actions: Sequence[str] = (),
    min_options: int = 2,
    drop_single_option: bool = True,
) -> ReferenceSet:
    """Group trace steps into a `ReferenceSet`, one trajectory per session.

    Steps whose option set has fewer than `min_options` entries carry no
    decision -- the policy over a single option is 1 by construction and
    contributes exactly zero to every distance -- so they are dropped by
    default rather than padding the reference set with steps that cannot move.
    """
    kept: dict[str, list[Step]] = {}
    dropped = 0
    for s in steps:
        options = list(dict.fromkeys(list(s.admissible) + list(extra_actions)))
        if len(options) < min_options:
            if drop_single_option:
                dropped += 1
                continue
        kept.setdefault(s.session, []).append(
            Step(
                observation=f"{s.role}: {len(s.messages)} messages",
                admissible=options,
                action=s.action,
                context=render(s.messages, skip_system=False),
                role=s.role,
                meta={"messages": list(s.messages), **s.meta},
            )
        )

    trajectories = [
        Trajectory(task=session, initial_observation=f"session {session}", steps=rows)
        for session, rows in kept.items()
        if rows
    ]
    if not trajectories:
        raise ValueError(
            "no usable steps: every decision had fewer than "
            f"{min_options} options (dropped {dropped})"
        )
    return ReferenceSet(trajectories=trajectories, generator=generator)


@dataclass(frozen=True)
class ChatHarness:
    """Rebuild a recorded prompt under a different configuration.

    This is where a configuration change enters. The conversation is replayed
    as the trace recorded it; what the configuration supplies is the system
    prompt and the block describing the available skills. Scoring the same
    reference steps under two of these answers "given the identical situation,
    does the new prompt decide differently".

    `system=None` leaves the recorded system message inside the body, which is
    the right setting for comparing models or temperatures with the prompt held
    fixed. Set it to compare prompts.
    """

    system: str | None = None
    skills: tuple[str, ...] | None = None
    skill_header: str = "Available skills:"
    cue: str = "Choose one option."

    def body(self, trajectory: Trajectory, t: int) -> str:
        step = trajectory.steps[t]
        messages = step.meta.get("messages")
        text = (
            render(messages, skip_system=self.system is not None)
            if messages
            else (step.context or "")
        )
        parts = [text]
        if self.skills is not None:
            parts.append("")
            parts.append(self.skill_header)
            parts += [f"- {s}" for s in self.skills]
        parts.append("")
        parts.append(self.cue)
        return "\n".join(parts)
