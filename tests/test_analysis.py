"""Composition, layout and the scope diagnosis."""

import numpy as np
import pytest
from conftest import toy_logit, toy_scorer

import lm_behavior as lb


def test_rank_preservation_is_stable_against_tied_distances():
    """A rectangle has three pairs of exactly equal distances.

    Reconstructed through an eigendecomposition they differ in the last bit, so
    an exact-equality tie rule reports 0.956 for a perfect recovery.
    """
    pts = np.array([[0.0, 0.0], [3.0, 0.0], [0.0, 4.0], [3.0, 4.0]])
    d = np.linalg.norm(pts[:, None, :] - pts[None, :, :], axis=-1)
    layout, _ = lb.mds(d)
    assert lb.rank_preservation(d, layout) == pytest.approx(1.0, abs=1e-9)


def test_mds_recovers_a_planar_configuration():
    rng = np.random.default_rng(11)
    pts = rng.normal(size=(7, 2))
    d = np.linalg.norm(pts[:, None, :] - pts[None, :, :], axis=-1)
    layout, retained = lb.mds(d)
    assert retained == pytest.approx(1.0, abs=1e-9)
    back = np.linalg.norm(layout[:, None, :] - layout[None, :, :], axis=-1)
    assert np.allclose(back, d, atol=1e-9)
    assert lb.rank_preservation(d, layout) == pytest.approx(1.0, abs=1e-9)


def test_procrustes_is_zero_for_a_rotated_copy():
    rng = np.random.default_rng(0)
    a = rng.normal(size=(6, 2))
    t = np.pi / 3
    rot = np.array([[np.cos(t), -np.sin(t)], [np.sin(t), np.cos(t)]])
    assert lb.procrustes(a, (a @ rot) * 2.5 + 7.0) == pytest.approx(0.0, abs=1e-9)


def test_composition_is_exact_when_the_components_do_not_interact(refs):
    """A change that is additive in the logits must compose exactly.

    Uses the unnormalised coordinate, which is the one that is linear in the
    scorer's output. Any departure from cosine 1 here would be a bug in the
    composition arithmetic rather than a property of the agents.
    """

    def make(shift: float):
        def fn(body, action, system, s=shift):
            # vary with the action itself; every action here is the same length
            return toy_logit(body, action, 42) + s * int(action.rsplit("_", 1)[1])

        return lb.FunctionScorer(fn=fn, name=f"add{shift}")

    base = lb.score_references(lb.Agent("base", make(0.0)), refs)
    only_a = lb.score_references(lb.Agent("a", make(0.5)), refs)
    only_b = lb.score_references(lb.Agent("b", make(0.3)), refs)
    both = lb.score_references(lb.Agent("ab", make(0.8)), refs)

    c = lb.compose_check(base, both, [only_a, only_b], alpha=1.0, normalized=False)
    assert c.cosine == pytest.approx(1.0, abs=1e-9)
    assert c.norm_ratio == pytest.approx(1.0, abs=1e-9)
    assert c.relative_error == pytest.approx(0.0, abs=1e-9)


def test_composition_through_the_softmax_is_not_exact(refs):
    """The same additive change stops composing once a softmax is in the way.

    This is not a defect of the tool. It is why the normalised coordinate and
    the unnormalised one are both carried, and why measured compositions report
    a direction that adds and a magnitude that does not.
    """

    def make(shift: float):
        def fn(body, action, system, s=shift):
            # vary with the action itself; every action here is the same length
            return toy_logit(body, action, 42) + s * int(action.rsplit("_", 1)[1])

        return lb.FunctionScorer(fn=fn, name=f"add{shift}")

    base = lb.score_references(lb.Agent("base", make(0.0)), refs)
    only_a = lb.score_references(lb.Agent("a", make(0.5)), refs)
    only_b = lb.score_references(lb.Agent("b", make(0.3)), refs)
    both = lb.score_references(lb.Agent("ab", make(0.8)), refs)

    c = lb.compose_check(base, both, [only_a, only_b], alpha=1.0, normalized=True)
    assert 0.5 < c.cosine < 1.0


def test_tangent_log_map_is_zero_at_the_base_point(vectors):
    v = lb.fisher_rao_log(vectors[0], vectors[0])
    assert float(np.linalg.norm(v)) == pytest.approx(0.0, abs=1e-12)


def test_tangent_log_map_recovers_the_arc_length(vectors):
    """||Log_b(p)|| is the geodesic distance, not the chord."""
    got = float(np.linalg.norm(lb.fisher_rao_log(vectors[1], vectors[0])))
    want = lb.distance(
        lb.embed(vectors[0], alpha=0.0), lb.embed(vectors[1], alpha=0.0), "fisher_rao"
    )
    assert got == pytest.approx(want, rel=1e-9)


def test_step_angles_stay_in_range(vectors):
    a = lb.step_angles(vectors[1], vectors[0])
    assert a.min() >= 0.0
    assert a.max() <= np.pi / 2 + 1e-9


def test_diagnose_warns_when_every_pair_is_saturated(refs):
    """Six unrelated policies on one map is the failure the paper names.

    The warning has to fire here: this is exactly the configuration that
    produced a rank correlation of 0.39 in the reproduction.
    """
    agents = [lb.Agent(f"m{i}", toy_scorer(100 + i)) for i in range(6)]
    d = lb.diagnose(lb.score_all(agents, refs))
    assert not d.ok
    assert any("span" in w or "maximum possible" in w for w in d.warnings)


def test_diagnose_reports_the_pair_count(vectors):
    d = lb.diagnose(vectors)
    assert d.n_agents == 3
    assert d.n_pairs == 3
    assert d.kl_max >= d.kl_median >= d.kl_min
    assert "agents" in str(d)


def test_diagnose_needs_two_agents(vectors):
    with pytest.raises(ValueError):
        lb.diagnose(vectors[:1])
