"""The drawings, and the honesty they are supposed to carry.

matplotlib is optional, so the whole module skips without it. The backend is
fixed to Agg before pyplot is imported: these tests must not open a window.
"""

import matplotlib
import numpy as np
import pytest

matplotlib.use("Agg")

import lm_behavior as lb  # noqa: E402

pytestmark = pytest.mark.filterwarnings("ignore::UserWarning")


@pytest.fixture(autouse=True)
def _no_leftover_figures():
    """Each test owns its own figures; savefig is exercised in its own test."""
    import matplotlib.pyplot as plt

    plt.close("all")
    yield
    plt.close("all")


def _texts(ax) -> list[str]:
    return [t.get_text() for t in ax.texts]


def _equidistant(n: int = 4) -> list[lb.Coordinates]:
    """n agents all the same distance apart: a configuration no plane holds."""
    return [
        lb.Coordinates(
            name=f"p{i}", z=np.eye(n)[i], alpha=1.0, n_steps=n, mean_length=4.0
        )
        for i in range(n)
    ]


def test_map_reports_the_same_numbers_as_the_layout_it_draws(vectors):
    coords = [lb.embed(v, alpha=0.0) for v in vectors]
    names, dist = lb.pairwise(coords)
    layout, retained = lb.mds(dist)

    drawn = lb.plot_map(coords)
    assert drawn.names == names
    assert drawn.retained == pytest.approx(retained)
    assert drawn.rank_preservation == pytest.approx(lb.rank_preservation(dist, layout))
    assert np.allclose(drawn.distance, dist)


def test_every_agent_is_labelled_so_identity_is_never_colour_alone(vectors):
    coords = [lb.embed(v, alpha=0.0) for v in vectors]
    drawn = lb.plot_map(coords)
    assert set(_texts(drawn.ax)) >= {c.name for c in coords}
    # the axes carry no readable scale: only the distances mean anything
    assert drawn.ax.get_aspect() == 1.0


def test_a_map_of_two_points_is_refused(vectors):
    coords = [lb.embed(v, alpha=0.0) for v in vectors][:2]
    with pytest.raises(ValueError, match="at least 3"):
        lb.plot_map(coords)


def test_a_lossy_plane_says_so_on_the_figure():
    """Four equidistant agents: a plane keeps 2 of the 3 equal dimensions."""
    drawn = lb.plot_map(_equidistant(4))
    assert drawn.retained == pytest.approx(2 / 3, abs=1e-9)
    assert not drawn.trustworthy
    assert "too lossy" in drawn.caption
    assert any("drawing, not a measurement" in t for t in _texts(drawn.ax))


def test_a_faithful_plane_carries_no_warning():
    rng = np.random.default_rng(3)
    pts = rng.normal(size=(6, 2))
    coords = [
        lb.Coordinates(name=f"p{i}", z=np.append(p, 0.0), alpha=1.0, n_steps=6,
                       mean_length=4.0)
        for i, p in enumerate(pts)
    ]
    drawn = lb.plot_map(coords)
    assert drawn.retained == pytest.approx(1.0, abs=1e-9)
    assert drawn.trustworthy
    assert not any("drawing, not a measurement" in t for t in _texts(drawn.ax))


def test_role_bars_repeat_the_indistinguishable_note(refs):
    """Two identical configurations: the bars are float noise, and say so."""
    from conftest import toy_scorer

    a = lb.score_references(lb.Agent("a", toy_scorer(1), lb.BaseHarness()), refs)
    b = lb.score_references(lb.Agent("b", toy_scorer(1), lb.BaseHarness()), refs)
    ax = lb.plot_by_role(lb.by_role(a, b))
    assert "indistinguishable" in ax.get_xlabel()


def test_role_bars_label_each_bar_with_its_value_and_share(vectors):
    contributions = lb.by_role(vectors[0], vectors[1])
    ax = lb.plot_by_role(contributions)
    assert "indistinguishable" not in ax.get_xlabel()
    assert len(ax.patches) == len(contributions)
    assert len(_texts(ax)) == len(contributions)
    assert "%" in "".join(_texts(ax))
    # per_step drops the share, which is not a share of anything per step
    per_step = lb.plot_by_role(contributions, per_step=True)
    assert "%" not in "".join(_texts(per_step))


def test_role_bars_refuse_an_empty_split():
    with pytest.raises(ValueError, match="no role contributions"):
        lb.plot_by_role([])


def test_the_heatmap_labels_every_cell_and_both_axes(vectors):
    coords = [lb.embed(v, alpha=0.0) for v in vectors]
    names, dist = lb.pairwise(coords)
    ax = lb.plot_distances(names, dist)
    assert [t.get_text() for t in ax.get_yticklabels()] == names
    assert len(_texts(ax)) == len(names) ** 2


def test_the_heatmap_refuses_a_matrix_that_does_not_match_the_names():
    with pytest.raises(ValueError, match="expected"):
        lb.plot_distances(["a", "b"], np.zeros((3, 3)))


def test_top_steps_draws_one_bar_per_step(vectors):
    steps = lb.top_steps(vectors[0], vectors[1], n=5)
    ax = lb.plot_top_steps(steps)
    assert len(ax.patches) == 5
    assert [t.get_text() for t in ax.get_yticklabels()][0].startswith("step ")
    with pytest.raises(ValueError, match="no steps"):
        lb.plot_top_steps([])


def test_save_figure_writes_a_file_and_closes_it(vectors, tmp_path):
    import matplotlib.pyplot as plt

    coords = [lb.embed(v, alpha=0.0) for v in vectors]
    drawn = lb.plot_map(coords)
    path = lb.save_figure(drawn.ax, str(tmp_path / "map.png"))
    assert (tmp_path / "map.png").stat().st_size > 1000
    assert path == str(tmp_path / "map.png")
    assert plt.get_fignums() == []


def test_cli_map_writes_the_image_it_was_asked_for(vectors, tmp_path, monkeypatch, capsys):
    """The --plot branch, with the scoring stubbed: it is the wiring under test."""
    import argparse

    from lm_behavior import cli

    monkeypatch.setattr(cli, "_load", lambda args: (None, vectors))
    out = tmp_path / "map.png"
    code = cli.cmd_map(
        argparse.Namespace(alpha=0.0, metric=None, out=None, plot=str(out))
    )
    assert code == 0
    assert out.stat().st_size > 1000
    printed = capsys.readouterr().out
    assert str(out) in printed
    assert "retains" in printed
