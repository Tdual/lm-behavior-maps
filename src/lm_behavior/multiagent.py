"""A service whose steps are decided by different components.

A router, a specialist and a synthesiser are three policies, not one. They run
different prompts, often on different models, and a change to one of them
should be attributable to that one. `MultiAgent` maps each step's `role` to the
component that owns it; everything downstream -- coordinates, distances, exact
KL -- is unchanged, because the reference steps are still the same steps.

What this can and cannot compare is decided by the reference set, not by this
class. Changing a prompt, a skill list or a model leaves the step sequence
alone, so both configurations score the same reference steps and the comparison
holds. Adding a component, or changing who runs when, changes the sequence
itself: the new configuration does not produce these steps at all, and no
amount of scoring will make the two comparable.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from .harness import BaseHarness, Harness
from .scorer import Scorer


@dataclass
class Role:
    """One component: its model, its prompt assembly, its sampling."""

    scorer: Scorer
    harness: Harness = field(default_factory=BaseHarness)
    temperature: float = 1.0

    @property
    def system(self) -> str | None:
        return getattr(self.harness, "system", None)


@dataclass
class MultiAgent:
    """One configuration of a multi-component service.

    `roles` is keyed by `Step.role`. `default` covers steps whose role is unset
    or unknown; without one, an unmapped role is an error rather than a silent
    omission, since dropping steps would change the coordinate for one
    configuration only and corrupt every distance involving it.
    """

    name: str
    roles: dict[str, Role]
    default: Role | None = None

    def for_role(self, role: str | None) -> Role:
        if role is not None and role in self.roles:
            return self.roles[role]
        if self.default is not None:
            return self.default
        known = ", ".join(sorted(self.roles)) or "none"
        raise KeyError(
            f"{self.name!r} has no component for role {role!r} (known: {known}). "
            "Add it to `roles`, or set `default`."
        )

    @property
    def fingerprint(self) -> str:
        parts = [
            f"{k}={v.scorer.fingerprint}@T{v.temperature}"
            for k, v in sorted(self.roles.items())
        ]
        return f"multi:{self.name}:" + "|".join(parts)
