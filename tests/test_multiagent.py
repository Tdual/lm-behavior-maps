"""A service whose steps belong to different components."""

import numpy as np
import pytest
from conftest import toy_scorer

import lm_behavior as lb


def service_refs() -> lb.ReferenceSet:
    """Three sessions of a router -> specialist -> synthesiser pipeline."""
    trajectories = []
    for s in range(3):
        steps = [
            lb.Step(
                observation=f"query {s}",
                admissible=["billing", "technical", "general"],
                action="technical",
                role="router",
            ),
            lb.Step(
                observation="specialist turn 1",
                admissible=["search", "fetch_doc", "calc", "__finish__"],
                action="search",
                role="specialist",
            ),
            lb.Step(
                observation="specialist turn 2",
                admissible=["search", "fetch_doc", "calc", "__finish__"],
                action="__finish__",
                role="specialist",
            ),
            lb.Step(
                observation="synthesis",
                admissible=["answer", "escalate"],
                action="answer",
                role="synth",
            ),
        ]
        trajectories.append(
            lb.Trajectory(task=f"session-{s}", initial_observation="start", steps=steps)
        )
    return lb.ReferenceSet(trajectories=trajectories, generator="test-service")


def config(name, router_seed=1, specialist_seed=2, synth_seed=3) -> lb.MultiAgent:
    return lb.MultiAgent(
        name=name,
        roles={
            "router": lb.Role(toy_scorer(router_seed, f"router{router_seed}")),
            "specialist": lb.Role(toy_scorer(specialist_seed, f"spec{specialist_seed}")),
            "synth": lb.Role(toy_scorer(synth_seed, f"synth{synth_seed}")),
        },
    )


def test_each_component_is_scored_by_its_own_model():
    refs = service_refs()
    v = lb.score_references(config("v1"), refs)
    assert v.n_steps == refs.n_steps
    assert v.roles == ["router", "specialist", "specialist", "synth"] * 3
    assert all(np.isclose(p.sum(), 1.0) for p in v.policies)


def test_steps_come_back_in_reference_order():
    """Grouping by role must not reorder the coordinate.

    The coordinate is defined by the order of the reference steps, so a
    misalignment here would silently corrupt every pairwise quantity.
    """
    refs = service_refs()
    v = lb.score_references(config("v1"), refs)
    sizes = [len(s.admissible) for _, _, _, s in refs.flat()]
    assert [len(p) for p in v.policies] == sizes
    assert [v.action_index[k] for k in range(4)] == [1, 0, 3, 0]


def test_a_single_agent_and_a_multi_agent_agree_when_all_roles_share_a_model():
    """MultiAgent must be a pure routing layer, not a different computation."""
    refs = service_refs()
    one = lb.score_references(lb.Agent("flat", toy_scorer(9, "same")), refs)
    many = lb.score_references(
        lb.MultiAgent(
            "split",
            roles={r: lb.Role(toy_scorer(9, "same")) for r in ("router", "specialist", "synth")},
        ),
        refs,
    )
    for a, b in zip(one.policies, many.policies, strict=True):
        assert np.allclose(a, b, atol=1e-12)


def test_an_unmapped_role_is_an_error_not_a_dropped_step():
    """Dropping steps would change the coordinate for one configuration only."""
    refs = service_refs()
    partial = lb.MultiAgent("gap", roles={"router": lb.Role(toy_scorer(1))})
    with pytest.raises(KeyError, match="synth|specialist"):
        lb.score_references(partial, refs)


def test_a_default_role_covers_the_gap():
    refs = service_refs()
    agent = lb.MultiAgent(
        "with-default",
        roles={"router": lb.Role(toy_scorer(1))},
        default=lb.Role(toy_scorer(2)),
    )
    assert lb.score_references(agent, refs).n_steps == refs.n_steps


def test_attribution_finds_the_component_that_changed():
    """Change only the specialist; the breakdown must point at the specialist."""
    refs = service_refs()
    v1 = lb.score_references(config("v1"), refs)
    v2 = lb.score_references(config("v2", specialist_seed=99), refs)

    rows = lb.by_role(v1, v2)
    assert rows[0].role == "specialist"
    changed = {r.role: r.total for r in rows}
    assert changed["router"] == pytest.approx(0.0, abs=1e-12)
    assert changed["synth"] == pytest.approx(0.0, abs=1e-12)
    assert changed["specialist"] > 0


def test_the_parts_add_up_to_the_whole():
    refs = service_refs()
    v1 = lb.score_references(config("v1"), refs)
    v2 = lb.score_references(config("v2", router_seed=7, specialist_seed=8), refs)
    rows = lb.by_role(v1, v2)
    assert sum(r.total for r in rows) == pytest.approx(lb.exact_symmetric_kl(v1, v2), rel=1e-9)
    assert sum(r.share for r in rows) == pytest.approx(1.0, rel=1e-9)
    assert sum(r.n_steps for r in rows) == v1.n_steps


def test_attribution_also_works_on_the_hellinger_metric():
    refs = service_refs()
    v1 = lb.score_references(config("v1"), refs)
    v2 = lb.score_references(config("v2", synth_seed=77), refs)
    rows = lb.by_role(v1, v2, metric="hellinger")
    assert rows[0].role == "synth"


def test_top_steps_points_at_individual_decisions():
    refs = service_refs()
    v1 = lb.score_references(config("v1"), refs)
    v2 = lb.score_references(config("v2", router_seed=55), refs)
    top = lb.top_steps(v1, v2, n=3)
    assert len(top) == 3
    assert top[0][1] == "router"
    assert top[0][2] >= top[-1][2]


def test_format_by_role_is_printable():
    refs = service_refs()
    v1 = lb.score_references(config("v1"), refs)
    v2 = lb.score_references(config("v2", specialist_seed=12), refs)
    text = lb.format_by_role(lb.by_role(v1, v2))
    assert "specialist" in text and "share" in text


def test_roles_must_match_between_two_runs():
    refs = service_refs()
    v1 = lb.score_references(config("v1"), refs)
    v2 = lb.score_references(config("v2"), refs)
    v2.roles = list(reversed(v2.roles))
    with pytest.raises(ValueError, match="different roles"):
        lb.by_role(v1, v2)


def test_two_configurations_do_not_trigger_the_spread_warning():
    """With one pair the spread is 1.0 by definition and means nothing.

    Comparing exactly two configurations is the most common use, so a warning
    that always fires there would train people to ignore the diagnosis.
    """
    refs = service_refs()
    v1 = lb.score_references(config("v1"), refs)
    v2 = lb.score_references(config("v2", specialist_seed=50), refs)
    d = lb.diagnose([v1, v2])
    assert not any("span only" in w for w in d.warnings)
    assert any("only 1 pairs" in w for w in d.warnings)
