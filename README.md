# lm-behavior-maps

Locate LLM agents by what they *would* do, without running them.

Each agent teacher-forces the actions of a shared set of reference
trajectories. The per-step log-likelihoods become its coordinates, and the
distance between two agents tracks the KL divergence between their policies.
No rollouts, no environment to step, no sampling — the cost is one forward pass
per (reference step, admissible action), linear in the number of agents.

An **agent** here is an LLM *plus a harness*: history window, system prompt,
instruction wording, sampling temperature, chat template. Two agents can share
every weight and still be different points on the map. That is the point.

This implements the method of *Agent Behavior Maps*
([arXiv:2610.30014](https://arxiv.org/abs/2610.30014)), with an
information-geometric re-derivation of the coordinate and a scope check the
paper does not provide. It grew out of a
[small reproduction](https://github.com/Tdual/agent-behavior-maps-repro) whose
measurements are quoted throughout this README and in the docstrings.

## Install

```bash
pip install lm-behavior-maps          # core: numpy only
pip install "lm-behavior-maps[hf]"    # + local HuggingFace weights
pip install "lm-behavior-maps[lmeval]" # + any lm-evaluation-harness model
```

## Use

```python
import lm_behavior as lb

refs   = lb.load_references("refs.json")
scorer = lb.HFScorer("Qwen/Qwen2.5-0.5B-Instruct")

agents = [
    lb.Agent("window-4", scorer, lb.BaseHarness(window=4)),
    lb.Agent("window-1", scorer, lb.BaseHarness(window=1)),
    lb.Agent("no-system", scorer, lb.BaseHarness(window=4, system=None)),
    lb.Agent("hot", scorer, lb.BaseHarness(window=4), temperature=1.25),
]

vectors = lb.score_all(agents, refs, cache=lb.ScoreCache())

print(lb.diagnose(vectors))        # before anything else: is this in scope?

coords     = lb.embed_all(vectors, alpha=0)
names, d   = lb.pairwise(coords)
layout, kept = lb.mds(d)
```

Or from the shell:

```bash
lm-behavior diagnose --refs refs.json --agents agents.json
lm-behavior map      --refs refs.json --agents agents.json --alpha 0 --out map.json
```

## Read `diagnose` first

The guarantees cover *close pairs*. When they do not hold, the failure is not
an error — it is a map that looks fine and means nothing.

In the reproduction, six different LLMs on one map gave a rank correlation of
**0.39**; nine harness variants of a single model gave **0.97**. The difference
was not the locality constant, which was *worse* in the run that worked. It was
whether the pairs spanned a range of true distances at all. When every pair
already sits near the maximum possible difference, there is no order left to
recover.

`diagnose` computes that from the policy vectors, before you trust anything:

```
4 agents, 6 pairs, 193 reference steps
  exact symmetric KL   min 0.0561  median 1.4510  max 32.6000
  spread (max/min)     581.1x
  saturation           0.29 of maximum entropy
  epsilon (median)     10.97
  no warnings
```

## What this delivers, and what it does not

| Question | Answer |
|---|---|
| Did this harness change move the agent, and how much relative to other changes? | **Yes.** Rank correlation 0.975 against measured KL on 36 pairs. |
| Is this quantized/re-templated/re-stacked build still the same agent? | **Yes**, and sensitively — dropping a chat template moved a model 86% as far as swapping it for a different model. |
| Which of these 100 agents are near-duplicates? | **Yes.** Coordinate neighbours have similar behaviour (edit 0.897, 2-gram 0.950). |
| How many nats apart are these two? | **No.** The `alpha=1` estimate overshoots by 3.5–4.2x. Use `alpha=0`, which held the level to 0.87x, or read the rank only. |
| Can I compare this map to one built last quarter? | **Only if the reference set is identical.** Swapping the reference *generator* moved a map 2–3x more than resampling from the same one. |
| Which agent is better? | **Not measured, and not measurable this way.** The method locates what an agent *would do*, not whether it succeeds. |

## Choosing alpha

Amari's α-representation writes a distribution as `p^((1-α)/2)`. By Chentsov's
theorem the α-connections are the *complete* list of invariant ones, so this is
not a menu of heuristics.

| α | coordinate | flat | metric | use it when |
|---|---|---|---|---|
| 1 | `log p` | yes (exponential families) | no | you need displacements to **add** — composing harness changes, Appendix-C style |
| 0 | `sqrt p` | no | yes (exact isometry to a sphere) | you want the **distance** to be right — default |
| −1 | `p` | yes (mixture families) | no | mixture-family work |

Measured on the same 36 pairs, against the exact reference KL:

| coordinate | Pearson | Spearman | level ratio |
|---|---|---|---|
| α=1 `log p` (unnormalised, the paper's) | 0.951 | 0.932 | 3.54 |
| α=1 `log p` (normalised over admissible) | 0.964 | 0.953 | 3.69 |
| α=0 `sqrt p` (Hellinger chord) | 0.991 | 0.983 | 0.82 |
| **α=0 `sqrt p` (Fisher–Rao arc)** | **0.994** | **0.990** | **0.87** |

α=0 never overestimated (0 of 36 pairs). The embedding error is essentially
solved there; what remains is the mismatch between the reference distribution
and where the agent actually goes, and no coordinate fixes that.

### The temperature blind spot

Two temperatures of one model share every token log-probability, so the paper's
unnormalised coordinate puts them at **exactly zero** distance while their KL
is not zero. `lb.embed(..., normalized=True)` and `alpha=0` both see it. The
package carries both conventions so the choice is measured, not assumed.

## Backends

The one primitive is `loglikelihood([(body, action)]) -> [log p]`, deliberately
the shape of `lm_eval.api.model.LM.loglikelihood`.

| Backend | Status |
|---|---|
| `HFScorer` — local weights via `transformers` | shipped |
| `LMEvalScorer` — wraps any lm-evaluation-harness model (HF, vLLM, compatible servers) | shipped |
| `FunctionScorer` — any callable | shipped |

**Hosted chat APIs generally cannot do this.** Teacher forcing needs the
log-probability of an action *you supply*. OpenAI's `echo` is gone from the
chat endpoint, and Anthropic exposes no log-probabilities. What works is local
weights, vLLM (`prompt_logprobs`), and self-hosted OpenAI-compatible servers.
Top-k truncation is not a problem: keeping only the top 50 logprobs reproduced
the full-vocabulary distances at Spearman 1.0000.

## The exact KL is free

The forward pass already scores every admissible action. Keeping the whole
policy vector gives the **exact** per-step KL for every pair, in closed form,
at the same number of forward passes — `lb.pairwise_exact_kl(vectors)`.

That is also the ceiling for everything else here: every reference-set quantity
approximates it, so how well it predicts on-policy KL bounds what any embedding
can reach. Measured, that ceiling was Spearman 0.931. An embedding scoring
*above* it is not tracking KL better; it is correlating with the mismatch.

## Development

```bash
python3.12 -m venv .venv && .venv/bin/pip install -e ".[dev]"
.venv/bin/pytest -q
.venv/bin/ruff check .
```

## Citation

The method is from *Agent Behavior Maps: Locating Agents by What They Would Do,
Without Running Them* (arXiv:2610.30014). This package is an independent
implementation; the authors' own code is not public.

## License

MIT.
