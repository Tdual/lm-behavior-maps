"""The one primitive every backend must provide.

    loglikelihood([(body, action), ...], system=...) -> [summed log p(action | body), ...]

This is deliberately the shape of `lm_eval.api.model.LM.loglikelihood`, minus
the `is_greedy` flag, so an lm-eval model can be adapted in a few lines and the
whole HF / vLLM / compatible-server ecosystem comes along with it.

What it demands is teacher forcing: the log-probability of an action *we
supply*, not of one the model generated. Hosted chat endpoints return
log-probabilities only for tokens they generated, so they cannot serve this
directly.

There is a second primitive that they can serve:

    policy(body, actions) -> array summing to one over `actions`

A backend that can return the whole next-token distribution gives the policy in
one call, without teacher forcing anything. That is how a hosted chat API gets
in: present the admissible set as single-token labels and read `top_logprobs`.
`Scorer` and `PolicyScorer` are both accepted; scoring prefers `policy` when a
backend offers it, since it also costs one call per step instead of one per
admissible action.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Protocol, runtime_checkable

import numpy as np


@runtime_checkable
class Scorer(Protocol):
    """Scores supplied continuations under supplied prompt bodies."""

    @property
    def fingerprint(self) -> str:
        """Stable identity of the weights plus decoding setup, for the cache."""
        ...

    def loglikelihood(
        self, pairs: Sequence[tuple[str, str]], *, system: str | None = None
    ) -> list[float]:
        """Summed token log-probability of each `action` given each `body`.

        `system` is applied the way the model expects: the system role when the
        model has a chat template, a plain prefix when it does not.
        """
        ...


@runtime_checkable
class PolicyScorer(Protocol):
    """Returns the whole policy over the admissible set in one call.

    Cheaper than scoring each action separately, and reachable from backends
    that expose a next-token distribution but cannot teacher-force. The
    unnormalised coordinate is not available this way -- there is no score
    outside the admissible set to be unnormalised with respect to -- which
    costs nothing, since that coordinate is the one that cannot see a
    temperature change.
    """

    @property
    def fingerprint(self) -> str: ...

    def policy(
        self, body: str, actions: Sequence[str], *, system: str | None = None
    ) -> np.ndarray: ...
