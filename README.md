# lm-behavior-maps

**Did that prompt change move your agent? Which part of it moved?**

Change a system prompt, add a skill, swap a model — then find out how far the
service's behaviour actually shifted, and which component shifted, **without
running the service once**.

```
role          steps      total   share   per step
specialist       48     1.8431  97.4%     0.0384
router           24     0.0492   2.6%     0.0021
synth            24     0.0000   0.0%     0.0000
(kl, symmetric, scaled to one session)
```

You edited the specialist's prompt. The router moved a little, the synthesiser
not at all. That took zero rollouts and no evaluation set.

It works by replaying the decision points of a recorded run: each
configuration scores the actions that were actually taken, and the distance
between the resulting coordinates tracks the KL divergence between the two
policies. Record once with the LangGraph or CrewAI adapter, then compare as
many configurations as you like for the cost of one forward pass per decision.

**What it answers is "how much did this change the behaviour", not "is it
better".** It is a regression detector, not a benchmark — a cheap filter to run
before the expensive evaluation, or to catch the change nobody intended.

This implements the method of *Agent Behavior Maps*
([arXiv:2610.30014](https://arxiv.org/abs/2610.30014)), with an
information-geometric re-derivation of the coordinate and a scope check the
paper does not provide. It grew out of a
[small reproduction](https://github.com/Tdual/agent-behavior-maps-repro) whose
measurements are quoted throughout this README and in the docstrings.

## Install

```bash
pip install lm-behavior-maps             # core: numpy only
pip install "lm-behavior-maps[hf]"        # + local HuggingFace weights
pip install "lm-behavior-maps[openai]"    # + hosted APIs (OpenAI, Gemini, Together, ...)
pip install "lm-behavior-maps[langgraph]" # + the LangGraph recorder
pip install "lm-behavior-maps[crewai]"    # + the CrewAI recorder
pip install "lm-behavior-maps[lmeval]"    # + any lm-evaluation-harness model
```

## A multi-agent service, end to end

```python
import lm_behavior as lb

# 1. record one run of the service you already have
recorder = lb.LangGraphRecorder(session="thread-1")
app.invoke(state, {"callbacks": [recorder]})
refs = recorder.references()

# 2. describe the configurations you want to compare
def configuration(name, specialist_prompt, skills):
    scorer = lb.ChoiceScorer("gpt-4.1-mini")     # or lb.HFScorer("...") locally
    return lb.MultiAgent(name, roles={
        "router":     lb.Role(scorer, lb.ChatHarness(system=ROUTER_PROMPT)),
        "specialist": lb.Role(scorer, lb.ChatHarness(system=specialist_prompt,
                                                     skills=skills)),
        "synth":      lb.Role(scorer, lb.ChatHarness(system=SYNTH_PROMPT)),
    })

old = configuration("v1", SPECIALIST_V1, ("search_kb", "fetch_doc"))
new = configuration("v2", SPECIALIST_V2, ("search_kb", "fetch_doc", "check_billing"))

# 3. score both against the same decision points
v1, v2 = lb.score_all([old, new], refs, cache=lb.ScoreCache())

# 4. read what moved
print(lb.format_by_role(lb.by_role(v1, v2)))
for index, role, value in lb.top_steps(v1, v2, n=5):
    print(f"step {index}  {role}  {value:.4f}")
```

`examples/service_regression.py` is this script, runnable with no API key.

### CrewAI

```python
recorder = lb.CrewAIRecorder(session="run-1")
with recorder:
    crew.kickoff(inputs=...)
refs = recorder.references()
```

### Anything else

Both adapters produce the same small contract, which is what to write against
if your framework is not one of them:

```python
steps = [lb.TraceStep(session="s1", role="router",
                      messages=[{"role": "user", "content": "..."}],
                      admissible=["search", "fetch", "__finish__"],
                      action="search")]
refs = lb.build_references(steps)
```

## What changes can be compared

A comparison holds when both configurations face the same decision points.

| Change | Comparable |
|---|---|
| system prompt, instruction wording | **yes** |
| skill or tool *description* | **yes** |
| model, temperature, sampling | **yes** |
| adding a tool | **partly** — measured as how preference over the existing options shifted; pass `extra_actions` to put the new one in front of both |
| adding a component, changing who runs when | **no** — the new configuration does not produce these steps at all |

Free-form prose is out of scope throughout: the method needs a finite, known
set of options at each step. In an agent service that set is the tool list and
the routing choice, which is where the behaviour lives anyway.

## Single agent

```python
refs   = lb.load_references("refs.json")
scorer = lb.HFScorer("Qwen/Qwen2.5-0.5B-Instruct")
agents = [lb.Agent("window-4", scorer, lb.BaseHarness(window=4)),
          lb.Agent("window-1", scorer, lb.BaseHarness(window=1))]

vectors    = lb.score_all(agents, refs, cache=lb.ScoreCache())
names, d   = lb.pairwise(lb.embed_all(vectors, alpha=0))
layout, _  = lb.mds(d)
```

From the shell:

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

Two primitives, because hosted endpoints can serve one of them but not the other.

**`loglikelihood([(body, action)]) -> [log p]`** — teacher forcing. Deliberately
the shape of `lm_eval.api.model.LM.loglikelihood`. The agent stays the agent:
nothing about the task is reformatted.

**`policy(body, actions) -> distribution`** — read the whole policy at once. No
teacher forcing, one call per step instead of one per action.

| Backend | Primitive | Runs on |
|---|---|---|
| `HFScorer` | teacher forcing | local weights via `transformers` |
| `LMEvalScorer` | teacher forcing | any lm-evaluation-harness model (HF, vLLM, …). Cross-checked against `HFScorer` on the same model and harness to **2.3e-05** |
| `EchoScorer` | teacher forcing | `/v1/completions` with `echo=true` — vLLM's server, providers that kept the endpoint |
| `ChoiceScorer` | policy | **any chat API with `top_logprobs`** — OpenAI, Azure, Gemini/Vertex, Together, Fireworks, OpenRouter, vLLM. Verified live against `gpt-4.1-mini`, `gpt-4o-mini` and `gpt-5.4-nano` |
| `FunctionScorer` | teacher forcing | any callable |

| Recorder | Captures from | Verified against |
|---|---|---|
| `LangGraphRecorder` | LangChain callbacks: node name, bound tools, `tool_calls`, and the bare-label replies a conditional edge branches on | langgraph 1.2.20, langchain-core 1.7.6, verified live with `gpt-4.1-mini` |
| `CrewAIRecorder` | the event bus: `LLMCallStartedEvent` + `LLMCallCompletedEvent`, paired by `call_id` | crewai 1.15.36 |

### Hosted chat APIs, via `ChoiceScorer`

A chat endpoint returns log-probabilities only for tokens it generated, so it
cannot teacher-force a supplied action. But ask it for exactly one token with
`top_logprobs`, label the admissible actions `A`, `B`, `C` …, and that single
distribution **is** the policy.

```python
import lm_behavior as lb

scorer = lb.ChoiceScorer("gpt-4.1-mini")                # or any OpenAI-compatible
scorer = lb.ChoiceScorer("llama-3.3-70b",
                         base_url="https://api.together.xyz/v1")

scorer = lb.ChoiceScorer("gpt-5.4-nano", top_logprobs=5,  # a reasoning model
                         reasoning_effort="none", max_completion_tokens=16)
```

What the endpoint allows is the real constraint, and it differs by model.
Measured against the OpenAI API on 2026-11-09:

| model family | `top_logprobs` | what it needs |
|---|---|---|
| `gpt-4.1`, `gpt-4o` | 20 | nothing — the defaults work |
| `gpt-5.x` reasoning | **5** | `reasoning_effort="none"` and `max_completion_tokens` above 1; reasoning tokens are spent out of the same budget, so the default of 1 returns no content at all. The cap also limits the admissible set to 5 actions |
| `gpt-5-mini` | — | refuses log-probabilities outright (HTTP 403) |

`ChoiceScorer` translates each of those three failures into the configuration
that fixes it rather than passing the raw endpoint error through.

Truncation is not what limits this. On a measured ablation over a 151,936-token
vocabulary, keeping only the top 20 alternatives reproduced the full-vocabulary
α=0 distances at rank correlation **0.9992**, and the top 50 at **1.0000**.

Two things it costs you:

- **It is a different harness, so it is a different agent.** The policy of a
  model answering "B" is not the policy of the same model writing
  `go to the kitchen`. Both are real agents and both make valid maps — but a
  letter-choice map and a free-form map cannot be compared with each other.
- **The admissible set must fit inside `top_logprobs`.** With more actions than
  the cap, `ChoiceScorer` raises rather than flooring the options it cannot
  observe.

**Anthropic exposes no log-probabilities at all**, so Claude cannot be placed on
a map by either route.

### Decisions a conditional edge turns on

A LangGraph router that asks the model for a bare label and branches on the
answer binds no tools, so nothing in the callback payload says what the options
were. Declare them and the decision is measured like any other:

```python
rec = lb.LangGraphRecorder(node_actions={"triage": ["billing", "technical"]})
```

The label is matched against the reply case-insensitively, whole word before
substring. A reply that names none of the labels — or two of them — is dropped
and counted in `skipped_unmatched`, because declaration order is not evidence
and a guess here would put a fabricated action into the reference set.

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
