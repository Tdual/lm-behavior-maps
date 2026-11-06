"""Record a CrewAI run, through its event bus.

`step_callback` is not enough: it reports the action but not the prompt that
produced it, and teacher forcing needs both. The event bus carries both halves.
Verified against crewai 1.15.36, where `crewai/llms/base_llm.py` emits:

  * `LLMCallStartedEvent` with `messages`, `tools`, `call_id`, and `agent_role`
    derived from the originating agent;
  * `LLMCallCompletedEvent` with `response` and the same `call_id`.

The choice is read from the completed event's `response` when it carries tool
calls. When it does not -- CrewAI serialises responses in several shapes --
the recorder falls back to the `ToolUsageStartedEvent` that follows for the
same agent, which is the actual dispatch and so cannot be wrong about which
tool ran. Steps that resolve to neither are recorded as finishing.

The bus dispatches on a thread pool, so handlers can run concurrently and out
of order. Pairing by `call_id` is order-free and safe. The tool-dispatch
fallback is not: it assumes the dispatch for an agent follows that agent's own
unresolved call, which holds for a sequential crew and can be wrong under
heavy parallelism. All shared state is taken under a lock either way, so a
concurrent run cannot corrupt the collection -- at worst a fallback lands on
the wrong step of the same agent.
"""

from __future__ import annotations

import threading
from typing import Any

from .trace import FINISH, TraceStep


def _tool_names(tools: Any) -> list[str]:
    names = []
    for entry in tools or []:
        if isinstance(entry, dict):
            fn = entry.get("function")
            name = (fn or {}).get("name") if isinstance(fn, dict) else entry.get("name")
            if name:
                names.append(str(name))
        elif hasattr(entry, "name"):
            names.append(str(entry.name))
    return names


def _messages(messages: Any) -> list[dict]:
    if isinstance(messages, str):
        return [{"role": "user", "content": messages}]
    out = []
    for m in messages or []:
        if isinstance(m, dict):
            out.append({"role": str(m.get("role", "user")), "content": m.get("content", "")})
        else:
            out.append({"role": "user", "content": str(m)})
    return out


def _tool_from_response(response: Any) -> str | None:
    """Dig a tool name out of whatever shape the response was serialised into."""
    if isinstance(response, dict):
        calls = response.get("tool_calls")
        if isinstance(calls, list) and calls:
            first = calls[0]
            if isinstance(first, dict):
                fn = first.get("function")
                name = (fn or {}).get("name") if isinstance(fn, dict) else first.get("name")
                if name:
                    return str(name)
        choices = response.get("choices")
        if isinstance(choices, list) and choices:
            message = choices[0].get("message") if isinstance(choices[0], dict) else None
            if isinstance(message, dict):
                return _tool_from_response(message)
    return None


class CrewAIRecorder:
    """Collect decisions from a crew run.

        recorder = CrewAIRecorder(session="run-1")
        with recorder:
            crew.kickoff(inputs=...)
        refs = recorder.references()

    Entering the context manager registers handlers on CrewAI's event bus and
    leaving it removes them, so a recorder never outlives the run it is for.
    """

    def __init__(self, session: str = "run", *, finish_action: str = FINISH):
        self.session = session
        self.finish_action = finish_action
        self.steps: list[TraceStep] = []
        self.skipped_no_tools = 0
        self._pending: dict[str, dict] = {}
        self._awaiting_tool: dict[str, int] = {}  # agent_role -> index into steps
        self._lock = threading.Lock()
        self._scope = None

    # -- event handlers ---------------------------------------------------

    def on_llm_start(self, event) -> None:
        tools = _tool_names(getattr(event, "tools", None))
        with self._lock:
            if not tools:
                self.skipped_no_tools += 1
                return
            self._pending[str(getattr(event, "call_id", ""))] = {
                "role": str(getattr(event, "agent_role", None) or "crew"),
                "messages": _messages(getattr(event, "messages", None)),
                "tools": tools,
                "task": getattr(event, "task_name", None),
            }

    def on_llm_end(self, event) -> None:
        with self._lock:
            started = self._pending.pop(str(getattr(event, "call_id", "")), None)
        if started is None:
            return
        options = list(started["tools"])
        if self.finish_action:
            options.append(self.finish_action)
        chosen = _tool_from_response(getattr(event, "response", None))
        if chosen is not None and chosen not in options:
            options.append(chosen)
        step = TraceStep(
            session=self.session,
            role=started["role"],
            messages=started["messages"],
            admissible=options,
            action=chosen if chosen is not None else self.finish_action,
            meta={
                "task": started["task"],
                "resolved_from": "response" if chosen is not None else "pending",
            },
        )
        with self._lock:
            self.steps.append(step)
            if chosen is None:
                self._awaiting_tool[started["role"]] = len(self.steps) - 1

    def on_tool_start(self, event) -> None:
        """Correct the most recent unresolved step for this agent.

        The dispatch is authoritative: whatever the response looked like, this
        is the tool that actually ran.
        """
        role = str(getattr(event, "agent_role", None) or "crew")
        name = str(getattr(event, "tool_name", "") or "")
        with self._lock:
            index = self._awaiting_tool.pop(role, None)
            if index is None or not name:
                return
            step = self.steps[index]
            if name not in step.admissible:
                step.admissible.append(name)
            step.action = name
            step.meta["resolved_from"] = "tool_usage"

    # -- registration -----------------------------------------------------

    def __enter__(self):
        from crewai.events import crewai_event_bus
        from crewai.events.types.llm_events import (
            LLMCallCompletedEvent,
            LLMCallStartedEvent,
        )
        from crewai.events.types.tool_usage_events import ToolUsageStartedEvent

        self._scope = crewai_event_bus.scoped_handlers()
        self._scope.__enter__()

        @crewai_event_bus.on(LLMCallStartedEvent)
        def _started(source, event):  # noqa: ARG001 - bus calls with both
            self.on_llm_start(event)

        @crewai_event_bus.on(LLMCallCompletedEvent)
        def _completed(source, event):  # noqa: ARG001
            self.on_llm_end(event)

        @crewai_event_bus.on(ToolUsageStartedEvent)
        def _tool(source, event):  # noqa: ARG001
            self.on_tool_start(event)

        return self

    def __exit__(self, *exc):
        if self._scope is not None:
            self._scope.__exit__(*exc)
            self._scope = None
        return False

    def references(self, **kwargs):
        from .trace import build_references

        kwargs.setdefault("generator", "crewai")
        return build_references(self.steps, **kwargs)
