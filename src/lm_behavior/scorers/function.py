"""Wrap any callable, for tests and for backends this package does not ship."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass


@dataclass
class FunctionScorer:
    """`fn(body, action, system) -> float`, the summed token log-probability."""

    fn: Callable[[str, str, str | None], float]
    name: str = "function"

    @property
    def fingerprint(self) -> str:
        return f"function:{self.name}"

    def loglikelihood(
        self, pairs: Sequence[tuple[str, str]], *, system: str | None = None
    ) -> list[float]:
        return [float(self.fn(body, action, system)) for body, action in pairs]
