"""Teacher forcing over HTTP, through the completions endpoint.

`echo=true` with `logprobs` returns the log-probability of the *prompt* tokens,
which is exactly teacher forcing: send prompt plus action, keep the tokens of
the action. Unlike the chat route this needs no reformatting of the task, so
the agent stays the agent.

It works against vLLM's OpenAI-compatible server and providers that kept the
completions endpoint. OpenAI's own chat models do not offer it. If a server
accepts the request but returns no prompt log-probabilities, that is reported
rather than quietly turned into a number.
"""

from __future__ import annotations

from collections.abc import Sequence


class EchoScorer:
    """Score supplied continuations with `echo=true` on `/v1/completions`."""

    def __init__(
        self,
        model: str,
        *,
        client=None,
        base_url: str | None = None,
        api_key: str | None = None,
        prefix_space: bool = True,
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
        self.prefix_space = prefix_space
        self.extra_body = extra_body or {}

    @property
    def fingerprint(self) -> str:
        return f"echo:{self.base_url or 'default'}:{self.model}:sp={int(self.prefix_space)}"

    def loglikelihood(
        self, pairs: Sequence[tuple[str, str]], *, system: str | None = None
    ) -> list[float]:
        return [self._one(body, action, system) for body, action in pairs]

    def _one(self, body: str, action: str, system: str | None) -> float:
        prefix = ((system + "\n\n") if system else "") + body
        continuation = (" " if self.prefix_space else "") + action
        response = self.client.completions.create(
            model=self.model,
            prompt=prefix + continuation,
            max_tokens=0,
            echo=True,
            logprobs=1,
            **self.extra_body,
        )
        lp = response.choices[0].logprobs
        if lp is None or lp.token_logprobs is None:
            raise RuntimeError(
                "the server returned no prompt log-probabilities. `echo` is unsupported "
                "here; use HFScorer, a vLLM server, or ChoiceScorer."
            )
        return _sum_tail(lp.token_logprobs, lp.text_offset, len(prefix))


def _sum_tail(token_logprobs, text_offset, split: int) -> float:
    """Sum the log-probabilities of tokens that start at or after `split`.

    `text_offset` gives each token's character position in the echoed prompt,
    which is how the continuation is located without re-tokenising. The first
    token has a null log-probability (nothing predicts it) and is skipped.
    """
    if text_offset is None:
        raise RuntimeError("the server returned no text_offset; cannot locate the continuation")
    total = 0.0
    for value, offset in zip(token_logprobs, text_offset, strict=True):
        if offset >= split and value is not None:
            total += float(value)
    return total
