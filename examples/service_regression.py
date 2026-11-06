"""Did changing the specialist's prompt move the service? Which part moved?

The shape of a real regression check:

  1. record one run of the service, to get reference decision points
  2. describe the old and the new configuration
  3. score both against the same reference points -- no rollouts
  4. read which component moved, and by how much

Runs with no API key: the graph uses a fake chat model to produce a trace, and
a small local model does the scoring. Swap in `lb.ChoiceScorer("gpt-5.5")` to
point it at a hosted API.

    python examples/service_regression.py
"""

from typing import Annotated, TypedDict

from langchain_core.language_models.fake_chat_models import GenericFakeChatModel
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.tools import tool
from langchain_core.utils.function_calling import convert_to_openai_tool
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages

import lm_behavior as lb

MODEL = "HuggingFaceTB/SmolLM2-135M-Instruct"

ROUTER_PROMPT = "Route the request to the right specialist."
SPECIALIST_V1 = "You are a support specialist. Answer from the knowledge base."
SPECIALIST_V2 = (
    "You are a support specialist. Always check billing records before answering, "
    "and prefer the internal documentation over the public knowledge base."
)
SKILLS_V1 = ("search_kb", "fetch_doc")
SKILLS_V2 = ("search_kb", "fetch_doc", "check_billing")


@tool
def search_kb(q: str) -> str:
    """Search the knowledge base."""
    return "kb hit"


@tool
def fetch_doc(doc_id: str) -> str:
    """Fetch an internal document."""
    return "doc body"


class GraphState(TypedDict):
    messages: Annotated[list, add_messages]


def record_trace(n_sessions: int = 4) -> lb.ReferenceSet:
    """Run the service and collect its decision points."""
    tools = [convert_to_openai_tool(t) for t in (search_kb, fetch_doc)]
    recorder = lb.LangGraphRecorder()

    for s in range(n_sessions):
        scripted = iter(
            [
                AIMessage(content="", tool_calls=[{"name": "search_kb", "args": {}, "id": "1"}]),
                AIMessage(content="", tool_calls=[{"name": "fetch_doc", "args": {}, "id": "2"}]),
                AIMessage(content="here is the answer"),
            ]
        )
        model = GenericFakeChatModel(messages=scripted).bind(tools=tools)

        def router(state, m=model):
            return {"messages": [m.invoke(state["messages"])]}

        def specialist(state, m=model):
            return {"messages": [m.invoke(state["messages"])]}

        def synth(state, m=model):
            return {"messages": [m.invoke(state["messages"])]}

        g = StateGraph(GraphState)
        for name, fn in (("router", router), ("specialist", specialist), ("synth", synth)):
            g.add_node(name, fn)
        g.add_edge(START, "router")
        g.add_edge("router", "specialist")
        g.add_edge("specialist", "synth")
        g.add_edge("synth", END)

        recorder.session = f"session-{s}"
        g.compile().invoke(
            {"messages": [HumanMessage(f"customer question {s}")]},
            {"callbacks": [recorder]},
        )

    return recorder.references()


def configuration(name: str, specialist_prompt: str, skills) -> lb.MultiAgent:
    scorer = lb.HFScorer(MODEL)  # or: lb.ChoiceScorer("gpt-5.5")
    return lb.MultiAgent(
        name=name,
        roles={
            "router": lb.Role(scorer, lb.ChatHarness(system=ROUTER_PROMPT)),
            "specialist": lb.Role(scorer, lb.ChatHarness(system=specialist_prompt, skills=skills)),
            "synth": lb.Role(scorer, lb.ChatHarness(system="Write the final answer.")),
        },
    )


def main() -> None:
    refs = record_trace()
    print(f"recorded {refs.n_steps} decision points over {refs.n_trajectories} sessions")

    old = configuration("v1", SPECIALIST_V1, SKILLS_V1)
    new = configuration("v2", SPECIALIST_V2, SKILLS_V2)

    with lb.ScoreCache() as cache:
        v1, v2 = lb.score_all([old, new], refs, cache=cache)

    print("\n" + str(lb.diagnose([v1, v2])))

    print("\nwhat moved")
    print(lb.format_by_role(lb.by_role(v1, v2)))

    print("\nthe decisions that moved most")
    for index, role, value in lb.top_steps(v1, v2, n=3):
        print(f"  step {index:>3}  {role:<12}{value:.4f}")

    total = lb.exact_symmetric_kl(v1, v2)
    print(f"\nwhole service: symmetric KL {total:.4f} per session, zero rollouts")


if __name__ == "__main__":
    main()
