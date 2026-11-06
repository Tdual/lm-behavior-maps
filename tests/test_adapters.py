"""The framework adapters, against the real libraries.

Verified against langgraph 1.2.20 / langchain-core 1.7.6 and crewai 1.15.36.
The LangGraph test runs an actual graph with a fake chat model, so it exercises
the real callback payloads. The CrewAI test emits the real event classes on the
real bus, so the field names are checked against the installed version even
though no model is called.
"""

from __future__ import annotations

from typing import Annotated, TypedDict

import numpy as np
import pytest
from conftest import toy_logit, toy_scorer

import lm_behavior as lb

# The graph state must live at module scope: langgraph resolves a TypedDict's
# annotations through module globals, not through a fixture's locals.
try:
    from langgraph.graph.message import add_messages

    class GraphState(TypedDict):
        messages: Annotated[list, add_messages]

except ImportError:  # pragma: no cover - the langgraph tests skip without it
    GraphState = None

# ------------------------------------------------------------------ generic


def test_trace_step_rejects_an_action_outside_the_options():
    with pytest.raises(ValueError, match="not among the options"):
        lb.TraceStep(session="s", role="r", messages=[], admissible=["a"], action="b")


def test_build_references_groups_by_session():
    steps = [
        lb.TraceStep("s1", "router", [{"role": "user", "content": "hi"}], ["a", "b"], "a"),
        lb.TraceStep("s1", "worker", [{"role": "user", "content": "hi"}], ["c", "d"], "d"),
        lb.TraceStep("s2", "router", [{"role": "user", "content": "yo"}], ["a", "b"], "b"),
    ]
    refs = lb.build_references(steps, generator="unit")
    assert refs.n_trajectories == 2
    assert refs.n_steps == 3
    roles = [s.role for _, _, _, s in refs.flat()]
    assert roles == ["router", "worker", "router"]


def test_single_option_steps_are_dropped():
    """A policy over one option is 1 by construction and cannot move."""
    steps = [
        lb.TraceStep("s", "r", [], ["only"], "only"),
        lb.TraceStep("s", "r", [], ["a", "b"], "a"),
    ]
    assert lb.build_references(steps).n_steps == 1


def test_everything_single_option_is_an_error_not_an_empty_map():
    with pytest.raises(ValueError, match="no usable steps"):
        lb.build_references([lb.TraceStep("s", "r", [], ["only"], "only")])


def test_extra_actions_puts_a_new_skill_in_front_of_both_configurations():
    steps = [lb.TraceStep("s", "r", [], ["search", "calc"], "search")]
    refs = lb.build_references(steps, extra_actions=["new_skill"])
    step = refs.trajectories[0].steps[0]
    assert step.admissible == ["search", "calc", "new_skill"]


def test_chat_harness_replaces_the_system_prompt():
    messages = [
        {"role": "system", "content": "OLD SYSTEM"},
        {"role": "user", "content": "the question"},
    ]
    refs = lb.build_references([lb.TraceStep("s", "r", messages, ["a", "b"], "a")])
    traj = refs.trajectories[0]

    kept = lb.ChatHarness().body(traj, 0)
    assert "OLD SYSTEM" in kept  # no override: the logged prompt is replayed as-is

    replaced = lb.ChatHarness(system="NEW SYSTEM").body(traj, 0)
    assert "OLD SYSTEM" not in replaced
    assert "the question" in replaced


def test_chat_harness_lists_the_skills_it_is_configured_with():
    refs = lb.build_references(
        [lb.TraceStep("s", "r", [{"role": "user", "content": "q"}], ["a", "b"], "a")]
    )
    body = lb.ChatHarness(system="S", skills=("search", "browse")).body(refs.trajectories[0], 0)
    assert "Available skills:" in body
    assert "- search" in body and "- browse" in body


def test_a_trace_drives_a_full_multi_agent_comparison():
    steps = []
    for s in range(3):
        steps.append(
            lb.TraceStep(f"s{s}", "router", [{"role": "user", "content": f"q{s}"}],
                         ["billing", "technical"], "technical")
        )
        for t in range(2):
            steps.append(
                lb.TraceStep(f"s{s}", "specialist",
                             [{"role": "user", "content": f"q{s}.{t}"}],
                             ["search", "calc", "__finish__"], "search")
            )
    refs = lb.build_references(steps, generator="unit")

    # ChatHarness hands the system prompt to the scorer, not into the body, so
    # the scorer has to read it -- a scorer that ignores `system` would show no
    # change at all, which is the behaviour, not a bug.
    def sysaware(seed, name):
        return lb.FunctionScorer(
            fn=lambda body, action, system, s=seed: toy_logit(
                f"{system}|{body}", action, s
            ),
            name=name,
        )

    def cfg(name, spec_prompt):
        return lb.MultiAgent(name, roles={
            "router": lb.Role(sysaware(1, "r"), lb.ChatHarness(system="ROUTE")),
            "specialist": lb.Role(sysaware(2, "s"), lb.ChatHarness(system=spec_prompt)),
        })

    v1, v2 = lb.score_all([cfg("v1", "BE BRIEF"), cfg("v2", "BE THOROUGH")], refs)
    rows = lb.by_role(v1, v2)
    # only the specialist's prompt changed, so only the specialist may move
    moved = {r.role: r.total for r in rows}
    assert moved["router"] == pytest.approx(0.0, abs=1e-12)
    assert moved["specialist"] > 0


# ---------------------------------------------------------------- langgraph


@pytest.fixture
def langgraph_app():
    pytest.importorskip("langgraph")
    pytest.importorskip("langchain_core")
    from langchain_core.language_models.fake_chat_models import GenericFakeChatModel
    from langchain_core.messages import AIMessage
    from langchain_core.tools import tool
    from langchain_core.utils.function_calling import convert_to_openai_tool
    from langgraph.graph import END, START, StateGraph

    @tool
    def search(q: str) -> str:
        """Search."""
        return "r"

    @tool
    def calc(e: str) -> str:
        """Calc."""
        return "42"

    model = GenericFakeChatModel(
        messages=iter(
            [
                AIMessage(content="", tool_calls=[{"name": "search", "args": {}, "id": "1"}]),
                AIMessage(content="done"),
            ]
        )
    )
    bound = model.bind(tools=[convert_to_openai_tool(t) for t in (search, calc)])

    S = GraphState

    def router(state):
        return {"messages": [bound.invoke(state["messages"])]}

    def worker(state):
        return {"messages": [bound.invoke(state["messages"])]}

    g = StateGraph(S)
    g.add_node("router", router)
    g.add_node("worker", worker)
    g.add_edge(START, "router")
    g.add_edge("router", "worker")
    g.add_edge("worker", END)
    return g.compile()


def test_langgraph_recorder_captures_nodes_options_and_choices(langgraph_app):
    from langchain_core.messages import HumanMessage

    rec = lb.LangGraphRecorder(session="thread-1")
    langgraph_app.invoke({"messages": [HumanMessage("hi")]}, {"callbacks": [rec]})

    assert [s.role for s in rec.steps] == ["router", "worker"]
    assert rec.steps[0].admissible == ["search", "calc", "__finish__"]
    assert rec.steps[0].action == "search"
    assert rec.steps[1].action == "__finish__"  # no tool call -> finished
    assert all(s.session == "thread-1" for s in rec.steps)
    assert rec.steps[0].messages[0] == {"role": "user", "content": "hi"}
    # the worker saw one more message than the router
    assert len(rec.steps[1].messages) == len(rec.steps[0].messages) + 1


def test_langgraph_recorder_builds_a_usable_reference_set(langgraph_app):
    from langchain_core.messages import HumanMessage

    rec = lb.LangGraphRecorder(session="t")
    langgraph_app.invoke({"messages": [HumanMessage("hi")]}, {"callbacks": [rec]})
    refs = rec.references()
    assert refs.generator == "langgraph"
    assert refs.n_steps == 2

    v = lb.score_references(
        lb.MultiAgent("v1", roles={
            "router": lb.Role(toy_scorer(1, "a")),
            "worker": lb.Role(toy_scorer(2, "b")),
        }),
        refs,
    )
    assert v.roles == ["router", "worker"]
    assert all(np.isclose(p.sum(), 1.0) for p in v.policies)


def test_langgraph_recorder_can_filter_to_specific_nodes(langgraph_app):
    from langchain_core.messages import HumanMessage

    rec = lb.LangGraphRecorder(include_nodes={"router"})
    langgraph_app.invoke({"messages": [HumanMessage("hi")]}, {"callbacks": [rec]})
    assert [s.role for s in rec.steps] == ["router"]


# ------------------------------------------------------------------- crewai


@pytest.fixture
def crew_events():
    pytest.importorskip("crewai")
    from crewai.events.types.llm_events import (
        LLMCallCompletedEvent,
        LLMCallStartedEvent,
        LLMCallType,
    )
    from crewai.events.types.tool_usage_events import ToolUsageStartedEvent

    def completed(**kw):
        kw.setdefault("call_type", LLMCallType.LLM_CALL)
        return LLMCallCompletedEvent(**kw)

    return LLMCallStartedEvent, completed, ToolUsageStartedEvent


def test_crewai_recorder_pairs_started_and_completed_by_call_id(crew_events):
    started, completed, _ = crew_events
    rec = lb.CrewAIRecorder(session="run-1")

    rec.on_llm_start(
        started(
            call_id="c1",
            messages=[{"role": "user", "content": "find the invoice"}],
            tools=[
                {"type": "function", "function": {"name": "search"}},
                {"type": "function", "function": {"name": "fetch"}},
            ],
            agent_role="researcher",
        )
    )
    rec.on_llm_end(
        completed(
            call_id="c1",
            messages=[],
            response={"tool_calls": [{"function": {"name": "search"}}]},
        )
    )

    assert len(rec.steps) == 1
    step = rec.steps[0]
    assert step.role == "researcher"
    assert step.admissible == ["search", "fetch", "__finish__"]
    assert step.action == "search"
    assert step.meta["resolved_from"] == "response"


def test_crewai_recorder_falls_back_to_the_tool_dispatch(crew_events):
    """When the response shape hides the tool, the dispatch is authoritative."""
    started, completed, tool_started = crew_events
    rec = lb.CrewAIRecorder()

    rec.on_llm_start(
        started(
            call_id="c1",
            messages=[{"role": "user", "content": "q"}],
            tools=[{"function": {"name": "search"}}, {"function": {"name": "fetch"}}],
            agent_role="researcher",
        )
    )
    rec.on_llm_end(completed(call_id="c1", messages=[], response="I will use a tool."))
    assert rec.steps[0].action == "__finish__"  # unresolved so far

    rec.on_tool_start(tool_started(tool_name="fetch", agent_role="researcher", tool_args={}))
    assert rec.steps[0].action == "fetch"
    assert rec.steps[0].meta["resolved_from"] == "tool_usage"


def test_crewai_recorder_ignores_calls_with_no_tools(crew_events):
    started, completed, _ = crew_events
    rec = lb.CrewAIRecorder()
    rec.on_llm_start(started(call_id="c1", messages=[], tools=None, agent_role="writer"))
    rec.on_llm_end(completed(call_id="c1", messages=[], response="prose"))
    assert rec.steps == []
    assert rec.skipped_no_tools == 1


def test_crewai_recorder_registers_on_the_real_event_bus(crew_events):
    """The decorator form and the scoped teardown must work on the real bus."""
    started, completed, _ = crew_events
    from crewai.events import crewai_event_bus

    def emit(event):
        # the bus dispatches on a thread pool and hands back a future
        future = crewai_event_bus.emit(None, event=event)
        if future is not None:
            future.result(timeout=10)

    rec = lb.CrewAIRecorder(session="bus")
    with rec:
        emit(
            started(
                call_id="c9",
                messages=[{"role": "user", "content": "q"}],
                tools=[{"function": {"name": "a"}}, {"function": {"name": "b"}}],
                agent_role="analyst",
            )
        )
        emit(
            completed(
                call_id="c9", messages=[], response={"tool_calls": [{"function": {"name": "b"}}]}
            )
        )
    assert len(rec.steps) == 1
    assert rec.steps[0].action == "b"

    # after the scope, further events must not be collected
    emit(started(call_id="c10", messages=[], tools=[], agent_role="x"))
    assert len(rec.steps) == 1
