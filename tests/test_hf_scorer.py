"""HFScorer against a naive implementation.

The scorer materialises only the logits at the continuation positions. That
slicing is the one piece of arithmetic here that can be silently wrong -- an
off-by-one in the window picks up the neighbouring token's distribution and
still returns a plausible number. So it is checked against the obvious
implementation, on a batch with deliberately ragged prefix and continuation
lengths, which is where padding and the kept window interact.

Skipped when torch, transformers or the hub are unavailable.
"""

import numpy as np
import pytest

import lm_behavior as lb

pytest.importorskip("torch")
pytest.importorskip("transformers")

MODEL = "sshleifer/tiny-gpt2"

PAIRS = [
    ("Task: find the mug\nAction:", "go to kitchen"),
    ("Task: find the mug\nAction:", "open the fridge now please"),
    ("Task: cool the apple\nObservation: you see a fridge\nAction:", "take apple"),
    ("short", "a"),
    ("a much much much longer prompt body than the others here\nAction:", "x y z"),
]


@pytest.fixture(scope="module")
def scorer():
    try:
        return lb.HFScorer(MODEL, device="cpu", batch_size=4)
    except Exception as exc:  # no network, no cache
        pytest.skip(f"cannot load {MODEL}: {exc}")


def naive(model_id, pairs):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(model_id).eval()
    out = []
    for body, action in pairs:
        pre = tok(body, add_special_tokens=True)["input_ids"]
        cont = tok(" " + action, add_special_tokens=False)["input_ids"]
        ids = torch.tensor([pre + cont])
        with torch.no_grad():
            logp = torch.log_softmax(model(input_ids=ids).logits.float(), dim=-1)
        out.append(sum(logp[0, len(pre) - 1 + k, t].item() for k, t in enumerate(cont)))
    return np.array(out)


def test_sliced_scoring_matches_the_naive_implementation(scorer):
    got = np.array(scorer.loglikelihood(PAIRS))
    assert np.abs(got - naive(MODEL, PAIRS)).max() < 1e-4


def test_batching_does_not_change_the_answer(scorer):
    one_at_a_time = np.array([scorer.loglikelihood([p])[0] for p in PAIRS])
    batched = np.array(scorer.loglikelihood(PAIRS))
    assert np.abs(one_at_a_time - batched).max() < 1e-4


def test_scores_are_negative_and_finite(scorer):
    got = np.array(scorer.loglikelihood(PAIRS))
    assert np.all(np.isfinite(got))
    assert np.all(got < 0)
