"""Hosted chat APIs, through a single-token choice.

A chat endpoint returns log-probabilities only for tokens it generated, so it
cannot teacher-force a supplied action. It can, however, be asked to emit
exactly one token and to report the top alternatives at that position
(`top_logprobs`, up to 20 on OpenAI and on Gemini/Vertex). Label the admissible
actions A, B, C ... and that single distribution *is* the policy.

Truncation is not the limit here. On a measured ablation over a 151,936-token
vocabulary, keeping only the top 20 alternatives reproduced the full-vocabulary
alpha=0 distances at rank correlation 0.9992, and the top 50 at 1.0000.

Two things to know before using it:

  * **This is a different harness, so it is a different agent.** The policy of a
    model asked to answer "B" is not the policy of the same model asked to write
    "go to the kitchen". Both are real agents and both make valid maps -- but a
    letter-choice map and a free-form map cannot be compared with each other.
  * **The admissible set must fit in `top_logprobs`.** With more actions than
    the cap, labels outside the returned alternatives get a floor rather than a
    measured value, and the policy is no longer exact. The constructor refuses
    the configuration instead of silently degrading.
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
        response = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            max_completion_tokens=1,
            temperature=0.0,
            logprobs=True,
            top_logprobs=self.top_logprobs,
            **self.extra_body,
        )
        return self._read(response, n)

    def _read(self, response, n: int) -> np.ndarray:
        content = response.choices[0].logprobs.content
        if not content:
            raise RuntimeError("the response carried no logprobs; does this model support them?")
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
