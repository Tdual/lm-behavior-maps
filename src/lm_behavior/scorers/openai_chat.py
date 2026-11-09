"""Hosted chat APIs, through a single-token choice.

A chat endpoint returns log-probabilities only for tokens it generated, so it
cannot teacher-force a supplied action. It can, however, be asked to emit
exactly one token and to report the top alternatives at that position
(`top_logprobs`), and that single distribution *is* the policy.

Truncation is not the limit here. On a measured ablation over a 151,936-token
vocabulary, keeping only the top 20 alternatives reproduced the full-vocabulary
alpha=0 distances at rank correlation 0.9992, and the top 50 at 1.0000.

What the endpoint allows is the limit. Measured against the OpenAI API on
2026-11-09:

  ===================  =============  ==========================================
  model family         top_logprobs   what it needs
  ===================  =============  ==========================================
  gpt-4.1, gpt-4o      20             nothing; the defaults here work
  gpt-5.x reasoning    5              `reasoning_effort="none"` and a token
                                      budget above 1 -- the reasoning tokens are
                                      spent out of `max_completion_tokens`, so
                                      the default of 1 returns no content at all
  gpt-5-mini           none           refuses logprobs outright (403)
  ===================  =============  ==========================================

So the defaults target the non-reasoning models. A reasoning model needs
`ChoiceScorer(model, top_logprobs=5, reasoning_effort="none",
max_completion_tokens=16)`, which also caps the admissible set at 5 actions.

Two further things to know before using it:

  * **This is a different harness, so it is a different agent.** The policy of a
    model asked to answer "B" is not the policy of the same model asked to write
    "go to the kitchen". Both are real agents and both make valid maps -- but a
    letter-choice map and a free-form map cannot be compared with each other.
  * **The admissible set must fit in `top_logprobs`.** With more actions than
    the cap, labels outside the returned alternatives get a floor rather than a
    measured value, and the policy is no longer exact. The constructor refuses
    the configuration instead of silently degrading. The endpoint may also
    return *fewer* alternatives than asked for, in which case the unreturned
    labels take the floor. That is the one degradation this class cannot see
    coming, so `diagnose` is the check that matters on hosted maps.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

LABELS = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"


class ChoiceScorer:
    """Read the policy off one token position of a chat completion.

    `client` is anything exposing `chat.completions.create(...)` -- the
    `openai` SDK pointed at OpenAI, Azure, Gemini's compatibility endpoint,
    Together, Fireworks, OpenRouter or a local vLLM server.
    """

    def __init__(
        self,
        model: str,
        *,
        client=None,
        base_url: str | None = None,
        api_key: str | None = None,
        top_logprobs: int = 20,
        max_actions: int | None = None,
        floor: float = -20.0,
        max_completion_tokens: int = 1,
        reasoning_effort: str | None = None,
        temperature: float | None = 0.0,
        extra_body: dict | None = None,
    ):
        if client is None:
            try:
                from openai import OpenAI
            except ImportError as exc:  # pragma: no cover - import guard
                raise ImportError(
                    "pass a client, or install the extra: pip install 'lm-behavior-maps[openai]'"
                ) from exc
            kwargs = {}
            if base_url:
                kwargs["base_url"] = base_url
            if api_key:
                kwargs["api_key"] = api_key
            client = OpenAI(**kwargs)
        self.client = client
        self.model = model
        self.base_url = base_url
        self.top_logprobs = top_logprobs
        self.max_actions = max_actions or top_logprobs
        self.floor = floor
        self.max_completion_tokens = max_completion_tokens
        self.reasoning_effort = reasoning_effort
        self.temperature = temperature
        self.extra_body = extra_body or {}

    @property
    def fingerprint(self) -> str:
        return f"choice:{self.base_url or 'default'}:{self.model}:k={self.top_logprobs}"

    def render(self, body: str, actions: Sequence[str]) -> str:
        lines = [body, "", "Options:"]
        lines += [f"{LABELS[i]}. {a}" for i, a in enumerate(actions)]
        lines.append("")
        lines.append("Answer with the single letter of the best option.")
        return "\n".join(lines)

    def policy(
        self, body: str, actions: Sequence[str], *, system: str | None = None
    ) -> np.ndarray:
        n = len(actions)
        if n > self.max_actions:
            raise ValueError(
                f"{n} admissible actions but top_logprobs is {self.top_logprobs}. "
                "The policy cannot be read exactly; reduce the admissible set, or use a "
                "teacher-forcing backend (HFScorer, EchoScorer)."
            )
        messages = ([{"role": "system", "content": system}] if system else []) + [
            {"role": "user", "content": self.render(body, actions)}
        ]
        kwargs = dict(
            model=self.model,
            messages=messages,
            max_completion_tokens=self.max_completion_tokens,
            logprobs=True,
            top_logprobs=self.top_logprobs,
            **self.extra_body,
        )
        if self.temperature is not None:
            kwargs["temperature"] = self.temperature
        if self.reasoning_effort is not None:
            kwargs["reasoning_effort"] = self.reasoning_effort
        try:
            response = self.client.chat.completions.create(**kwargs)
        except Exception as exc:  # re-raised with the fix, never swallowed
            raise self._explain(exc) from exc
        return self._read(response, n)

    def _explain(self, exc: Exception) -> Exception:
        """Turn the three failures the OpenAI API actually returns into advice."""
        text = str(exc)
        if "top_logprobs" in text and "less than or equal to" in text:
            return ValueError(
                f"{self.model} caps top_logprobs below the requested {self.top_logprobs} "
                "(the gpt-5 family allows 5). Construct with top_logprobs=5, which also "
                f"caps the admissible set at 5 actions. Endpoint said: {text}"
            )
        if "logprobs" in text and ("not allowed" in text or "not supported" in text):
            return ValueError(
                f"{self.model} refuses log-probabilities, so it cannot be placed on a map "
                "by this route. Use a non-reasoning model (gpt-4.1, gpt-4o), or, for a "
                f"gpt-5 model, reasoning_effort='none'. Endpoint said: {text}"
            )
        if "max_tokens" in text or "output limit" in text:
            return ValueError(
                f"{self.model} spent the whole token budget before emitting a choice. "
                "A reasoning model needs reasoning_effort='none' and "
                "max_completion_tokens above 1, because reasoning tokens come out of "
                f"the same budget. Endpoint said: {text}"
            )
        return exc

    def _read(self, response, n: int) -> np.ndarray:
        content = response.choices[0].logprobs.content if response.choices[0].logprobs else None
        if not content:
            raise RuntimeError(
                "the response carried no logprobs. A reasoning model needs "
                "reasoning_effort='none' and max_completion_tokens above 1; otherwise the "
                f"budget is spent before any content token. finish_reason="
                f"{response.choices[0].finish_reason!r}"
            )
        alternatives = {}
        for item in content[0].top_logprobs:
            token = item.token.strip()
            if token and token[0] in LABELS[:n]:
                index = LABELS.index(token[0])
                # the same letter can arrive with and without a leading space
                alternatives[index] = np.logaddexp(
                    alternatives.get(index, -np.inf), float(item.logprob)
                )
        if not alternatives:
            raise RuntimeError(
                "none of the option letters appeared in top_logprobs; the model is not "
                "answering in the requested format"
            )
        logp = np.full(n, self.floor, dtype=np.float64)
        for index, value in alternatives.items():
            logp[index] = value
        p = np.exp(logp - logp.max())
        return p / p.sum()
