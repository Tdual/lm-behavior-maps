"""The two hosted-API backends, against reconstructed server responses.

No network: a live OpenAI or vLLM endpoint is not available here. What *is*
available is the exact response shape each endpoint documents, so the parsing
is driven by responses built to that shape -- and, for the echo backend, by a
response reconstructed from a real tokenizer and checked against HFScorer,
which is what actually exercises the character-offset arithmetic.
"""

from __future__ import annotations

import math
from types import SimpleNamespace

import numpy as np
import pytest

import lm_behavior as lb

# --------------------------------------------------------------- chat / choice


def chat_response(pairs):
    """pairs: [(token, logprob)] as `top_logprobs` would return them."""
    content = [SimpleNamespace(top_logprobs=[SimpleNamespace(token=t, logprob=v)
                                             for t, v in pairs])]
    return SimpleNamespace(choices=[SimpleNamespace(logprobs=SimpleNamespace(content=content))])


class FakeChat:
    def __init__(self, pairs):
        self.pairs = pairs
        self.calls = []
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self._create))

    def _create(self, **kwargs):
        self.calls.append(kwargs)
        return chat_response(self.pairs)


def test_choice_scorer_reads_the_policy_off_top_logprobs():
    want = np.array([0.6, 0.3, 0.1])
    client = FakeChat([(ch, math.log(p)) for ch, p in zip("ABC", want, strict=True)])
    s = lb.ChoiceScorer("gpt-x", client=client)
    got = s.policy("Task: do the thing", ["go north", "go south", "wait"])
    assert np.allclose(got, want, atol=1e-9)


def test_choice_scorer_renormalises_over_the_admissible_set():
    """Probability mass on anything that is not an option is dropped."""
    client = FakeChat([("A", math.log(0.2)), ("B", math.log(0.2)), ("Sure", math.log(0.6))])
    s = lb.ChoiceScorer("gpt-x", client=client)
    got = s.policy("body", ["x", "y"])
    assert np.allclose(got, [0.5, 0.5], atol=1e-9)


def test_choice_scorer_merges_a_letter_seen_with_and_without_a_space():
    client = FakeChat([("A", math.log(0.25)), (" A", math.log(0.25)), ("B", math.log(0.5))])
    s = lb.ChoiceScorer("gpt-x", client=client)
    assert np.allclose(s.policy("body", ["x", "y"]), [0.5, 0.5], atol=1e-9)


def test_choice_scorer_asks_for_one_token_with_logprobs():
    client = FakeChat([("A", -0.1), ("B", -2.0)])
    lb.ChoiceScorer("gpt-x", client=client, top_logprobs=20).policy("body", ["x", "y"])
    call = client.calls[0]
    assert call["max_completion_tokens"] == 1
    assert call["logprobs"] is True
    assert call["top_logprobs"] == 20


def test_choice_scorer_refuses_more_actions_than_it_can_observe():
    """Silently flooring unobserved options would corrupt the policy."""
    client = FakeChat([("A", -0.1)])
    s = lb.ChoiceScorer("gpt-x", client=client, top_logprobs=3)
    with pytest.raises(ValueError, match="top_logprobs"):
        s.policy("body", [f"act{i}" for i in range(5)])


def test_choice_scorer_reports_a_model_that_ignores_the_format():
    client = FakeChat([("Sure", -0.1), ("!", -2.0)])
    with pytest.raises(RuntimeError, match="not\n?\\s*answering|format"):
        lb.ChoiceScorer("gpt-x", client=client).policy("body", ["x", "y"])


def test_choice_scorer_drives_the_whole_pipeline(refs):
    """A PolicyScorer must reach coordinates and distances like any other."""

    class Rotating(FakeChat):
        def __init__(self, shift):
            super().__init__([])
            self.shift = shift

        def _create(self, **kwargs):
            self.calls.append(kwargs)
            n = kwargs["messages"][-1]["content"].count("\n") % 3
            weights = np.roll([0.5, 0.3, 0.15, 0.05], (n + self.shift) % 4)
            return chat_response(
                [(ch, math.log(w)) for ch, w in zip("ABCD", weights, strict=True)]
            )

    agents = [
        lb.Agent("p0", lb.ChoiceScorer("m", client=Rotating(0))),
        lb.Agent("p1", lb.ChoiceScorer("m", client=Rotating(1))),
    ]
    # distinct fingerprints, or the cache would merge the two agents
    agents[1].scorer.model = "m2"

    vectors = lb.score_all(agents, refs)
    assert all(np.isclose(p.sum(), 1.0) for v in vectors for p in v.policies)
    assert lb.exact_symmetric_kl(*vectors) > 0
    d = lb.distance(*lb.embed_all(vectors, alpha=0))
    assert d > 0
    assert vectors[0].meta["unnormalised_coordinate"] is False


# ------------------------------------------------------------- echo / teacher


def echo_response(token_logprobs, text_offset):
    return SimpleNamespace(
        choices=[
            SimpleNamespace(
                logprobs=SimpleNamespace(
                    token_logprobs=token_logprobs, text_offset=text_offset
                )
            )
        ]
    )


class FakeEcho:
    def __init__(self, fn):
        self.fn = fn
        self.calls = []
        self.completions = SimpleNamespace(create=self._create)

    def _create(self, **kwargs):
        self.calls.append(kwargs)
        return self.fn(kwargs["prompt"])


def test_echo_scorer_sums_only_the_continuation():
    prompt_len = len("hello world")

    def fn(prompt):
        assert prompt == "hello world yes"
        # four tokens: "hello", " world", " yes" plus the unpredicted first
        return echo_response([None, -1.0, -2.0, -4.0], [0, 5, prompt_len, prompt_len + 4])

    s = lb.EchoScorer("m", client=FakeEcho(fn))
    # tokens starting at or after len("hello world") are the continuation
    assert s.loglikelihood([("hello world", "yes")])[0] == pytest.approx(-6.0)


def test_echo_scorer_requests_echo_with_no_generation():
    s = lb.EchoScorer("m", client=FakeEcho(lambda p: echo_response([None, -1.0], [0, 1])))
    s.loglikelihood([("a", "b")])
    call = s.client.calls[0]
    assert call["echo"] is True
    assert call["max_tokens"] == 0
    assert call["logprobs"] == 1


def test_echo_scorer_reports_a_server_that_ignores_echo():
    s = lb.EchoScorer("m", client=FakeEcho(lambda p: echo_response(None, None)))
    with pytest.raises(RuntimeError, match="echo"):
        s.loglikelihood([("a", "b")])


def test_echo_scorer_matches_hf_on_a_reconstructed_response():
    """Build the response a real server would send, from a real tokenizer.

    This is the test that exercises the character-offset arithmetic: the split
    between prompt and continuation is found from `text_offset`, so a tokenizer
    that merges across the boundary, or an off-by-one in the comparison, shows
    up as a disagreement with HFScorer on the same text.
    """
    torch = pytest.importorskip("torch")
    pytest.importorskip("transformers")
    from transformers import AutoModelForCausalLM, AutoTokenizer

    model_id = "sshleifer/tiny-gpt2"
    try:
        hf = lb.HFScorer(model_id, device="cpu")
        tok = AutoTokenizer.from_pretrained(model_id)
        model = AutoModelForCausalLM.from_pretrained(model_id).eval()
    except Exception as exc:
        pytest.skip(f"cannot load {model_id}: {exc}")

    def server(prompt: str):
        ids = tok(prompt, add_special_tokens=True)["input_ids"]
        with torch.no_grad():
            logp = torch.log_softmax(
                model(input_ids=torch.tensor([ids])).logits.float(), dim=-1
            )
        token_logprobs = [None] + [
            logp[0, k, t].item() for k, t in enumerate(ids[1:])
        ]
        offsets, at = [], 0
        for t in ids:
            offsets.append(at)
            at += len(tok.decode([t]))
        return echo_response(token_logprobs, offsets)

    echo = lb.EchoScorer(model_id, client=FakeEcho(server))
    cases = [
        ("Task: find the mug\nAction:", "go to kitchen"),
        ("Task: cool the apple\nObservation: a fridge\nAction:", "take apple"),
    ]
    got = np.array(echo.loglikelihood(cases))
    want = np.array(hf.loglikelihood(cases))
    assert np.abs(got - want).max() < 1e-3


class RaisingChat:
    """A client whose endpoint refuses the request, with a real error message."""

    def __init__(self, message):
        self.message = message
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self._create))

    def _create(self, **kwargs):
        raise RuntimeError(self.message)


# The three refusals below are the verbatim messages the OpenAI API returned on
# 2026-11-09 for gpt-5.4-nano and gpt-5-mini. Each one is a configuration
# mistake with a known fix, so the scorer must name the fix.
@pytest.mark.parametrize(
    ("message", "expected"),
    [
        ("Invalid value for 'top_logprobs': must be less than or equal to 5.",
         "top_logprobs=5"),
        ("You are not allowed to request logprobs from this model",
         "reasoning_effort='none'"),
        ("Could not finish the message because max_tokens or model output limit was reached.",
         "max_completion_tokens above 1"),
    ],
)
def test_choice_scorer_translates_the_endpoints_refusals(message, expected):
    s = lb.ChoiceScorer("gpt-x", client=RaisingChat(message))
    with pytest.raises(ValueError, match=r".*") as excinfo:
        s.policy("body", ["x", "y"])
    assert expected in str(excinfo.value)
    assert message in str(excinfo.value)  # the raw endpoint text is never swallowed


def test_choice_scorer_passes_the_reasoning_knobs_through():
    client = FakeChat([("A", math.log(1.0))])
    s = lb.ChoiceScorer("gpt-5.4-nano", client=client, top_logprobs=5,
                        reasoning_effort="none", max_completion_tokens=16)
    s.policy("body", ["x", "y"])
    call = client.calls[-1]
    assert call["top_logprobs"] == 5
    assert call["reasoning_effort"] == "none"
    assert call["max_completion_tokens"] == 16


def test_choice_scorer_omits_temperature_when_asked_to():
    """Some endpoints reject `temperature` outright."""
    client = FakeChat([("A", math.log(1.0))])
    lb.ChoiceScorer("gpt-x", client=client, temperature=None).policy("body", ["x", "y"])
    assert "temperature" not in client.calls[-1]
