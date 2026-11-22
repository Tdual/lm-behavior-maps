"""Draw the figures the README shows, from the runnable example's own data.

Everything here is measured, not mocked: the same recorded LangGraph trace as
`service_regression.py`, scored by a small local model. Four configurations,
so there is a map to draw -- two would only have a distance.

    python examples/draw_figures.py            # writes docs/img/*.png
    python examples/draw_figures.py some/dir
"""

import sys
from pathlib import Path

from service_regression import (
    SKILLS_V1,
    SKILLS_V2,
    SPECIALIST_V1,
    SPECIALIST_V2,
    configuration,
    make_scorer,
    record_trace,
)

import lm_behavior as lb

TERSE = "You are a support specialist. Answer in one line."

HERE = Path(__file__).resolve().parent


def main() -> None:
    out = Path(sys.argv[1]) if len(sys.argv) > 1 else HERE.parent / "docs" / "img"
    out.mkdir(parents=True, exist_ok=True)

    refs = record_trace()
    scorer = make_scorer()
    configurations = [
        configuration("v1", SPECIALIST_V1, SKILLS_V1, scorer),
        configuration("v2", SPECIALIST_V2, SKILLS_V2, scorer),
        configuration("v2-no-billing", SPECIALIST_V2, SKILLS_V1, scorer),
        configuration("terse", TERSE, SKILLS_V1, scorer),
    ]
    with lb.ScoreCache() as cache:
        vectors = lb.score_all(configurations, refs, cache=cache)
    v1, v2 = vectors[0], vectors[1]
    # the role split is drawn for the pair that actually moved: rewriting the
    # specialist's prompt wholesale, not the two-word edit between v1 and v2
    terse = vectors[3]

    coords = lb.embed_all(vectors, alpha=0)
    # a wide, short axes: the points here sit in a horizontal band, and an
    # equal-aspect map on a square figure would be mostly empty
    import matplotlib.pyplot as plt

    _, ax = plt.subplots(figsize=(7.0, 3.4))
    drawn = lb.plot_map(coords, title="four configurations of one service", ax=ax)
    print(lb.save_figure(drawn.ax, str(out / "map.png")), "--", drawn.caption)

    names, dist = lb.pairwise(coords)
    print(lb.save_figure(lb.plot_distances(names, dist), str(out / "distances.png")))
    print(lb.save_figure(lb.plot_by_role(lb.by_role(v1, terse)), str(out / "roles.png")))
    print(lb.save_figure(lb.plot_top_steps(lb.top_steps(v1, terse, n=6)), str(out / "steps.png")))

    print("\nv1 against v2 (the two-word edit)")
    print(lb.format_by_role(lb.by_role(v1, v2)))
    print("\nv1 against terse (the rewrite, drawn in roles.png)")
    print(lb.format_by_role(lb.by_role(v1, terse)))


if __name__ == "__main__":
    main()
