"""Bridge to any `lm_eval.api.model.LM`.

lm-evaluation-harness already maintains adapters for HuggingFace, vLLM,
OpenAI-compatible servers and more, all behind one `loglikelihood` call. Rather
than re-implement them, wrap them -- the request shape is the same.

Two mismatches are handled here and worth knowing about:

  * lm-eval returns `(logprob, is_greedy)`; only the log-probability is used.
  * lm-eval has no system-prompt slot, so a `system` is prepended to the body.
    A model whose deployed form puts the system prompt in a chat role will sit
    somewhere slightly different than it would under `HFScorer`. Keep one
    backend for one map.
"""

from __future__ import annotations

from collections.abc import Sequence


class LMEvalScorer:
    """Adapt an lm-eval model object to the `Scorer` protocol."""

    def __init__(self, lm, name: str | None = None):
        if not hasattr(lm, "loglikelihood"):
            raise TypeError("expected an lm_eval LM instance with a loglikelihood method")
        self.lm = lm
        self.name = name or type(lm).__name__

    @property
    def fingerprint(self) -> str:
        model = getattr(self.lm, "model_args", None) or getattr(self.lm, "pretrained", "")
        return f"lmeval:{self.name}:{model}"

    def loglikelihood(
        self, pairs: Sequence[tuple[str, str]], *, system: str | None = None
    ) -> list[float]:
        bodies = [((system + "\n\n") if system else "") + b for b, _ in pairs]
        requests = [_Instance((ctx, cont)) for ctx, (_, cont) in zip(bodies, pairs, strict=True)]
        results = self.lm.loglikelihood(requests)
        return [float(r[0]) if isinstance(r, (tuple, list)) else float(r) for r in results]


class _Instance:
    """The minimal duck-type of `lm_eval.api.instance.Instance`."""

    __slots__ = ("args",)

    def __init__(self, args: tuple[str, str]):
        self.args = args
