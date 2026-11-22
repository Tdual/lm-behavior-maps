"""Drawings of a map, and the numbers that say whether to trust one.

matplotlib is an optional dependency (`pip install "lm-behavior-maps[viz]"`),
imported lazily so the core package stays numpy-only.

Every function here carries the honesty of the module it draws:

  * `plot_map` writes the retained fraction and the rank preservation into the
    caption, and says on the figure itself when the plane is too lossy to read
    as a measurement. A two-dimensional picture of a configuration that only
    half survives the projection is a drawing, and a drawing that looks like a
    measurement is worse than no picture at all;
  * the map's axes are equal-aspect and unlabelled, because only the distances
    between points carry meaning -- the orientation and the position of the
    origin are arbitrary products of the eigendecomposition;
  * `plot_by_role` repeats the note `format_by_role` prints when the two
    configurations are indistinguishable, so a bar chart of float noise cannot
    be mistaken for a finding.

Identity is never carried by colour alone: points and bars are labelled.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

from .attribution import NEGLIGIBLE, RoleContribution
from .distance import pairwise
from .embed import Coordinates
from .map2d import mds, rank_preservation

# Below this the plane has dropped so much of the configuration that the
# picture cannot be read as a statement about distances. The measured maps that
# this threshold comes from sat at 0.84-0.99; the 0.55 case in the paper's own
# figures is what it is meant to catch.
LOW_RETAINED = 0.8

MARK = "#1f4e79"  # one ink for the data; identity comes from the labels
WARN = "#b3261e"


def _fmt(value: float) -> str:
    """Four significant digits.

    A fixed number of decimals hides the magnitude of these quantities: a real
    per-role contribution can sit at 2e-04 as easily as at 5.8, and `0.0002`
    reads as a rounding artefact where `0.0001954` reads as a measurement.
    """
    return f"{value:.4g}"


def _pyplot():
    try:
        import matplotlib.pyplot as plt
    except ImportError as exc:  # pragma: no cover - depends on the environment
        raise ImportError(
            "drawing needs matplotlib, which is an optional dependency: "
            'pip install "lm-behavior-maps[viz]"'
        ) from exc
    return plt


def _axes(ax, size):
    plt = _pyplot()
    if ax is not None:
        return ax
    return plt.subplots(figsize=size)[1]


def _recede(ax) -> None:
    """Grid and frame to the background, where they belong."""
    ax.grid(True, color="#dddddd", linewidth=0.6)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color("#999999")
    ax.tick_params(colors="#666666", labelsize=9, length=3)


@dataclass
class MapPlot:
    """A drawn map, with the numbers needed to read it."""

    names: list[str]
    layout: np.ndarray
    distance: np.ndarray
    retained: float
    rank_preservation: float
    ax: object

    @property
    def trustworthy(self) -> bool:
        return self.retained >= LOW_RETAINED

    @property
    def caption(self) -> str:
        note = "" if self.trustworthy else "  -- too lossy to read as distances"
        return (
            f"2d layout retains {self.retained:.1%} of the configuration, "
            f"rank preservation {self.rank_preservation:.3f}{note}"
        )


def plot_map(
    coords: Sequence[Coordinates],
    *,
    metric: str | None = None,
    ax=None,
    title: str | None = None,
) -> MapPlot:
    """Scatter the agents on the MDS plane, labelled, with the caption.

    `coords` is what `embed_all` returns. The distances between the points are
    the quantity; the axes are not, so they are drawn equal-aspect and left
    without ticks.
    """
    _pyplot()  # fail on the missing dependency before doing the work
    if len(coords) < 3:
        raise ValueError(
            f"a map needs at least 3 agents to place; got {len(coords)}. "
            "For two, the distance itself is the whole result"
        )
    names, dist = pairwise(coords, metric=metric)
    layout, retained = mds(dist)
    rank = rank_preservation(dist, layout)

    ax = _axes(ax, (6.0, 5.2))
    ax.scatter(layout[:, 0], layout[:, 1], s=46, color=MARK, zorder=3)
    for (x, y), name in zip(layout, names, strict=True):
        # a fixed offset in points: the data units here have no scale of their own
        ax.annotate(
            name, (x, y), xytext=(0, 8), textcoords="offset points",
            ha="center", va="bottom", fontsize=10, color="#333333",
            annotation_clip=False,
        )
    # room for the labels above the topmost points, and for the outermost ones
    ax.margins(0.16, 0.20)
    ax.set_aspect("equal", adjustable="datalim")
    _recede(ax)
    ax.tick_params(labelbottom=False, labelleft=False, length=0)
    ax.set_title(title or "behaviour map", fontsize=12, color="#333333", loc="left")

    plot = MapPlot(names, layout, dist, retained, rank, ax)
    ax.set_xlabel(
        plot.caption, fontsize=9, color=MARK if plot.trustworthy else WARN, loc="left"
    )
    if not plot.trustworthy:
        ax.text(
            0.5, 0.5, "the plane is a drawing, not a measurement",
            transform=ax.transAxes, ha="center", va="center",
            fontsize=11, color=WARN, alpha=0.5, zorder=2,
        )
    return plot


def plot_distances(
    names: Sequence[str], matrix: np.ndarray, *, ax=None, title: str | None = None
) -> object:
    """The pairwise matrix as a single-hue heatmap, every cell labelled.

    Read this instead of the map when the layout's retained fraction is low:
    the matrix is the measurement, the plane is its projection.
    """
    _pyplot()
    matrix = np.asarray(matrix, dtype=np.float64)
    n = len(names)
    if matrix.shape != (n, n):
        raise ValueError(f"matrix is {matrix.shape}, expected ({n}, {n})")

    ax = _axes(ax, (0.9 * n + 2.2, 0.9 * n + 1.6))
    top = float(matrix.max()) or 1.0
    ax.imshow(matrix, cmap="Blues", vmin=0.0, vmax=top)
    ax.set_xticks(range(n), names, rotation=45, ha="right", fontsize=9)
    ax.set_yticks(range(n), names, fontsize=9)
    for i in range(n):
        for j in range(n):
            value = matrix[i, j]
            ax.text(
                j, i, _fmt(value), ha="center", va="center", fontsize=8,
                # the dark end of the ramp needs light ink
                color="#ffffff" if value > 0.6 * top else "#333333",
            )
    ax.grid(False)
    ax.tick_params(length=0)
    for side in ax.spines:
        ax.spines[side].set_visible(False)
    ax.set_title(title or "pairwise distance", fontsize=12, color="#333333", loc="left")
    return ax


def plot_by_role(
    contributions: Sequence[RoleContribution],
    *,
    metric: str = "kl",
    per_step: bool = False,
    ax=None,
) -> object:
    """Horizontal bars of what each component contributed.

    `per_step=True` draws the mean per step instead of the total, which is the
    comparison to make when the roles own very different numbers of steps.
    """
    _pyplot()
    if not contributions:
        raise ValueError("nothing to draw: no role contributions")
    rows = list(contributions)
    values = [c.per_step if per_step else c.total for c in rows]
    labels = [f"{c.role}  ({c.n_steps})" for c in rows]
    y = np.arange(len(rows))[::-1]  # largest first, reading downwards

    ax = _axes(ax, (6.4, 0.6 * len(rows) + 1.8))
    ax.barh(y, values, height=0.44, color=MARK)
    ax.set_yticks(y, labels, fontsize=10)
    total = sum(c.total for c in rows)
    widest = max(values) or 1.0
    for yi, c, v in zip(y, rows, values, strict=True):
        text = _fmt(v) if per_step else f"{_fmt(v)}   {c.share:.1%}"
        ax.text(v + 0.02 * widest, yi, text, va="center", fontsize=9, color="#333333")
    ax.set_xlim(0, widest * 1.32)
    _recede(ax)
    ax.grid(axis="y", visible=False)
    kind = "mean per step" if per_step else "total"
    ax.set_title(f"what moved, by role ({kind})", fontsize=12, color="#333333", loc="left")
    caption = f"{metric}, symmetric, scaled to one trajectory; (n) is the role's step count"
    if total < NEGLIGIBLE:
        caption = (
            f"the configurations are indistinguishable here (total < {NEGLIGIBLE:g}); "
            "these bars are noise"
        )
    ax.set_xlabel(
        caption, fontsize=9, color="#333333" if total >= NEGLIGIBLE else WARN, loc="left"
    )
    return ax


def plot_top_steps(
    steps: Sequence[tuple[int, str, float]], *, metric: str = "kl", ax=None
) -> object:
    """Horizontal bars of the individual reference steps that moved most."""
    _pyplot()
    if not steps:
        raise ValueError("nothing to draw: no steps")
    values = [v for _, _, v in steps]
    labels = [f"step {k}  {role}" for k, role, _ in steps]
    y = np.arange(len(steps))[::-1]

    ax = _axes(ax, (6.4, 0.42 * len(steps) + 1.8))
    ax.barh(y, values, height=0.42, color=MARK)
    ax.set_yticks(y, labels, fontsize=9)
    widest = max(values) or 1.0
    for yi, v in zip(y, values, strict=True):
        ax.text(v + 0.02 * widest, yi, _fmt(v), va="center", fontsize=9, color="#333333")
    ax.set_xlim(0, widest * 1.22)
    _recede(ax)
    ax.grid(axis="y", visible=False)
    ax.set_title("the steps that moved most", fontsize=12, color="#333333", loc="left")
    ax.set_xlabel(f"per-step {metric}, symmetric", fontsize=9, color="#333333", loc="left")
    return ax


def save_figure(ax, path: str, *, dpi: int = 160) -> str:
    """Write the figure an axes belongs to, tight, and close it."""
    plt = _pyplot()
    fig = ax.get_figure()
    fig.savefig(path, dpi=dpi, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return path
