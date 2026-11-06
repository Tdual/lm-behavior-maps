"""Command line: score a set of agents and print or save a map.

    lm-behavior diagnose --refs refs.json --agents agents.json
    lm-behavior map      --refs refs.json --agents agents.json --alpha 0 --out map.json

`agents.json` is a list of harness variants over local models:

    {"agents": [
      {"name": "w4", "model": "Qwen/Qwen2.5-0.5B-Instruct", "window": 4},
      {"name": "w1", "model": "Qwen/Qwen2.5-0.5B-Instruct", "window": 1},
      {"name": "hot", "model": "Qwen/Qwen2.5-0.5B-Instruct", "temperature": 1.25}
    ]}

One scorer is built per distinct (model, chat, dtype), so variants of one model
load the weights once.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

from .cache import ScoreCache
from .diagnostics import diagnose
from .distance import pairwise, to_pairs
from .embed import embed_all
from .harness import DEFAULT_SYSTEM, BaseHarness
from .kl import pairwise_exact_kl
from .map2d import mds, rank_preservation
from .references import load as load_references
from .scoring import Agent, score_all


def _build_agents(spec: dict):
    from .scorers import HFScorer

    scorers: dict[tuple, object] = {}
    agents = []
    for a in spec["agents"]:
        key = (a["model"], a.get("chat"), a.get("dtype", "float32"))
        if key not in scorers:
            scorers[key] = HFScorer(
                a["model"], chat=a.get("chat"), dtype=a.get("dtype", "float32")
            )
        harness = BaseHarness(
            window=a.get("window", 4),
            system=a.get("system", DEFAULT_SYSTEM) if a.get("system", True) else None,
        )
        agents.append(
            Agent(
                name=a["name"],
                scorer=scorers[key],
                harness=harness,
                temperature=float(a.get("temperature", 1.0)),
            )
        )
    return agents


def _load(args):
    refs = load_references(args.refs)
    spec = json.loads(Path(args.agents).read_text(encoding="utf-8"))
    agents = _build_agents(spec)
    cache = ScoreCache(args.cache, enabled=not args.no_cache)
    vectors = score_all(agents, refs, cache=cache, progress=True)
    cache.close()
    return refs, vectors


def cmd_diagnose(args) -> int:
    _, vectors = _load(args)
    d = diagnose(vectors)
    print(d)
    return 0 if d.ok else 1


def cmd_map(args) -> int:
    _, vectors = _load(args)
    print(diagnose(vectors), file=sys.stderr)

    coords = embed_all(vectors, alpha=args.alpha)
    names, dist = pairwise(coords, metric=args.metric)
    _, est = pairwise(coords, metric=args.metric, estimate_kl=True)
    _, exact = pairwise_exact_kl(vectors)
    layout, retained = mds(dist)

    out = {
        "names": names,
        "alpha": args.alpha,
        "metric": args.metric,
        "distance": dist.tolist(),
        "kl_estimate": est.tolist(),
        "exact_reference_kl": exact.tolist(),
        "layout_2d": layout.tolist(),
        "layout_retained": retained,
        "layout_rank_preservation": rank_preservation(dist, layout),
    }
    if args.out:
        Path(args.out).write_text(json.dumps(out, indent=1), encoding="utf-8")
        print(f"wrote {args.out}")
    else:
        width = max(len(n) for n in names) + 2
        print("\npairwise distance")
        print(" " * width + "".join(f"{n:>10}" for n in names))
        for i, n in enumerate(names):
            print(f"{n:<{width}}" + "".join(f"{dist[i, j]:>10.4f}" for j in range(len(names))))
        print(f"\n2d layout retains {retained:.1%} of the configuration, "
              f"rank preservation {out['layout_rank_preservation']:.3f}")
        print("\nclosest pairs by estimated symmetric KL")
        for (a, b), v in sorted(to_pairs(names, est).items(), key=lambda kv: kv[1])[:5]:
            print(f"  {a} / {b}: {v:.4f}")
    return 0


def main(argv=None) -> int:
    p = argparse.ArgumentParser(prog="lm-behavior", description=__doc__.split("\n")[0])
    p.add_argument("--version", action="store_true")
    sub = p.add_subparsers(dest="cmd")

    for name, fn in (("diagnose", cmd_diagnose), ("map", cmd_map)):
        s = sub.add_parser(name)
        s.add_argument("--refs", required=True)
        s.add_argument("--agents", required=True)
        s.add_argument("--cache", default=None)
        s.add_argument("--no-cache", action="store_true")
        s.set_defaults(fn=fn)
        if name == "map":
            s.add_argument("--alpha", type=float, default=0.0, choices=[1.0, 0.0, -1.0])
            s.add_argument(
                "--metric", default=None, choices=["euclidean", "hellinger", "fisher_rao"]
            )
            s.add_argument("--out", default=None)

    args = p.parse_args(argv)
    if args.version:
        from . import __version__

        print(__version__)
        return 0
    if not getattr(args, "fn", None):
        p.print_help()
        return 2
    np.set_printoptions(suppress=True)
    return args.fn(args)


if __name__ == "__main__":
    raise SystemExit(main())
