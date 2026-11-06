"""The one primitive every backend must provide.

    loglikelihood([(body, action), ...], system=...) -> [summed log p(action | body), ...]

This is deliberately the shape of `lm_eval.api.model.LM.loglikelihood`, minus
the `is_greedy` flag, so an lm-eval model can be adapted in a few lines and the
whole HF / vLLM / compatible-server ecosystem comes along with it.

What it demands is teacher forcing: the log-probability of an action *we supply*,
not of one the model generated. Hosted chat APIs generally cannot do this --
OpenAI's `echo` is gone from the chat endpoint and Anthropic exposes no
log-probabilities at all -- so the usable backends are local weights, vLLM
(`prompt_logprobs`), and self-hosted OpenAI-compatible servers.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Protocol, runtime_checkable


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
