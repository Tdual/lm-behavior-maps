"""Place four harness variants of one model, with no rollouts at all.

    python examples/make_refs.py
    python examples/quickstart.py

Downloads a 135M model on first run. Everything after that is cached.
"""

import lm_behavior as lb

MODEL = "HuggingFaceTB/SmolLM2-135M-Instruct"


def main() -> None:
    refs = lb.load_references("examples/refs.json")
    scorer = lb.HFScorer(MODEL)

    agents = [
        lb.Agent("window-4", scorer, lb.BaseHarness(window=4)),
        lb.Agent("window-1", scorer, lb.BaseHarness(window=1)),
        lb.Agent("no-system", scorer, lb.BaseHarness(window=4, system=None)),
        lb.Agent("hot", scorer, lb.BaseHarness(window=4), temperature=1.25),
    ]

    with lb.ScoreCache() as cache:
        vectors = lb.score_all(agents, refs, cache=cache, progress=True)

    print("\n" + str(lb.diagnose(vectors)))

    coords = lb.embed_all(vectors, alpha=0)
    names, dist = lb.pairwise(coords, metric="fisher_rao")
    _, exact = lb.pairwise_exact_kl(vectors)

    print("\n  pair                        distance   est. KL   exact KL")
    est = lb.to_pairs(names, lb.pairwise(coords, metric="fisher_rao", estimate_kl=True)[1])
    for (a, b), d in sorted(lb.to_pairs(names, dist).items(), key=lambda kv: kv[1]):
        i, j = names.index(a), names.index(b)
        print(f"  {a + ' / ' + b:<26}{d:9.4f}{est[(a, b)]:10.4f}{exact[i, j]:11.4f}")

    layout, retained = lb.mds(dist)
    print(f"\n  2-D layout retains {retained:.1%}, "
          f"rank preservation {lb.rank_preservation(dist, layout):.3f}")

    # The paper's unnormalised coordinate cannot separate temperatures.
    raw = lb.embed_all(vectors, alpha=1, normalized=False)
    print(f"\n  window-4 vs hot, unnormalised alpha=1: "
          f"{lb.distance(raw[0], raw[3]):.6f}  <- the temperature blind spot")


if __name__ == "__main__":
    main()
