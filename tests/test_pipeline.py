"""Scoring, caching, embedding and the temperature blind spot, end to end."""

import numpy as np
import pytest
from conftest import toy_scorer

import lm_behavior as lb


def test_identical_agents_have_zero_distance_and_zero_kl(refs):
    a = lb.score_references(lb.Agent("a", toy_scorer(7)), refs)
    b = lb.score_references(lb.Agent("b", toy_scorer(7)), refs)
    assert lb.exact_symmetric_kl(a, b) == pytest.approx(0.0, abs=1e-12)
    for alpha in (1.0, 0.0):
        za, zb = lb.embed(a, alpha=alpha), lb.embed(b, alpha=alpha)
        assert lb.distance(za, zb) == pytest.approx(0.0, abs=1e-12)


def test_policies_sum_to_one(vectors):
    for v in vectors:
        for p in v.policies:
            assert float(p.sum()) == pytest.approx(1.0, abs=1e-12)


def test_temperature_blind_spot_of_the_unnormalised_coordinate(refs):
    """Two temperatures of one model share every token log-probability.

    The paper's unnormalised coordinate therefore cannot tell them apart, while
    the KL between them is not zero. This is the reason the package carries
    both conventions rather than assuming one.
    """
    scorer = toy_scorer(11)
    cold = lb.score_references(lb.Agent("cold", scorer, temperature=0.8), refs)
    hot = lb.score_references(lb.Agent("hot", scorer, temperature=1.25), refs)

    raw = lb.distance(
        lb.embed(cold, alpha=1.0, normalized=False), lb.embed(hot, alpha=1.0, normalized=False)
    )
    assert raw == pytest.approx(0.0, abs=1e-12)

    norm = lb.distance(lb.embed(cold, alpha=1.0), lb.embed(hot, alpha=1.0))
    alpha0 = lb.distance(lb.embed(cold, alpha=0.0), lb.embed(hot, alpha=0.0))
    exact = lb.exact_symmetric_kl(cold, hot)
    assert exact > 0
    assert norm > 0
    assert alpha0 > 0


def test_alpha0_estimate_does_not_overshoot_the_exact_reference_kl(vectors):
    """The theory says the chord is a lower bound on KL; check the direction."""
    for i in range(len(vectors)):
        for j in range(i + 1, len(vectors)):
            est = lb.symmetric_kl_estimate(
                lb.embed(vectors[i], alpha=0.0), lb.embed(vectors[j], alpha=0.0), "hellinger"
            )
            exact = lb.exact_symmetric_kl(vectors[i], vectors[j])
            assert est <= exact * 2.0 + 1e-9  # the factor-4 estimator may exceed 1x


def test_cache_prevents_a_second_scorer_call(refs, tmp_path):
    calls = {"n": 0}

    def fn(body, action, system):
        calls["n"] += 1
        return float(len(body) % 7) - 3.0

    scorer = lb.FunctionScorer(fn=fn, name="counting")
    agent = lb.Agent("x", scorer)
    cache = lb.ScoreCache(tmp_path / "c.sqlite3")

    lb.score_references(agent, refs, cache=cache)
    first = calls["n"]
    assert first > 0

    lb.score_references(agent, refs, cache=cache)
    assert calls["n"] == first  # every value served from disk

    cache.close()


def test_cache_separates_different_scorers(refs, tmp_path):
    cache = lb.ScoreCache(tmp_path / "c.sqlite3")
    a = lb.score_references(lb.Agent("a", toy_scorer(1)), refs, cache=cache)
    b = lb.score_references(lb.Agent("b", toy_scorer(2)), refs, cache=cache)
    cache.close()
    assert lb.exact_symmetric_kl(a, b) > 0


def test_harness_window_changes_the_coordinate(refs):
    scorer = toy_scorer(5)
    wide = lb.score_references(lb.Agent("w4", scorer, lb.BaseHarness(window=4)), refs)
    narrow = lb.score_references(lb.Agent("w0", scorer, lb.BaseHarness(window=0)), refs)
    assert lb.exact_symmetric_kl(wide, narrow) > 0


def test_pairwise_matrix_is_symmetric_with_zero_diagonal(vectors):
    names, d = lb.pairwise(lb.embed_all(vectors, alpha=0.0))
    assert names == ["a", "b", "c"]
    assert np.allclose(d, d.T)
    assert np.allclose(np.diag(d), 0.0)


def test_mismatched_alpha_is_refused(vectors):
    with pytest.raises(ValueError):
        lb.distance(lb.embed(vectors[0], alpha=1.0), lb.embed(vectors[1], alpha=0.0))


def test_reference_roundtrip(refs, tmp_path):
    p = tmp_path / "refs.json"
    lb.save_references(refs, p)
    back = lb.load_references(p)
    assert back.n_steps == refs.n_steps
    assert back.generator == "test-fixture"
    assert back.trajectories[0].steps[0].admissible == refs.trajectories[0].steps[0].admissible


def test_step_rejects_an_action_outside_the_admissible_set():
    with pytest.raises(ValueError):
        lb.Step(observation="o", admissible=["a", "b"], action="c")

def test_continuation_never_doubles_the_separator():
    """An action handed in with its own leading space must not get a second one.

    Measured on SmolLM2-135M: the double space moved the log-likelihood 4.4
    nats away from the joint tokenisation that lm-eval and a from-scratch
    reference both compute.
    """
    from types import SimpleNamespace

    from lm_behavior.scorers.hf import HFScorer

    on = SimpleNamespace(prefix_space=True)
    off = SimpleNamespace(prefix_space=False)
    assert HFScorer._continuation(on, "go north") == " go north"
    assert HFScorer._continuation(on, " go north") == " go north"
    assert HFScorer._continuation(on, "\tgo north") == "\tgo north"
    assert HFScorer._continuation(off, "go north") == "go north"
