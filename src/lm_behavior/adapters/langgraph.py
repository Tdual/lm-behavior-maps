"""Record a LangGraph run, through LangChain's callback interface.

Attaching here rather than to LangGraph's own internals is deliberate: the
callback protocol and `AIMessage.tool_calls` belong to `langchain-core` and
move slowly, while graph internals do not. Verified against langgraph 1.2.20
and langchain-core 1.7.6, where:

  * `on_chat_model_start` carries the conversation, `metadata["langgraph_node"]`
    (the node that is running) and `invocation_params["tools"]` (the options);
  * `on_llm_end` carries the choice as `tool_calls` on the generated message,
    but *not* the metadata -- so the two halves are paired by `run_id`.

A node that does not call a model, or calls one with no tools bound, records
nothing: there is no policy over a finite set to measure there.
"""

from __future__ import annotations

from typing import Any

from .trace import FINISH, TraceStep

try:  # the handler is only useful with langchain-core present
    from langchain_core.callbacks import BaseCallbackHandler
except ImportError:  # pragma: no cover - import guard
    BaseCallbackHandler = object  # type: ignore[assignment,misc]


def _tool_names(invocation_params: dict | None) -> list[str]:
    """Option names from the OpenAI-shaped tool list LangChain passes along."""
    names = []
    for entry in (invocation_params or {}).get("tools") or []:
        if isinstance(entry, dict):
            fn = entry.get("function")
            name = (fn or {}).get("name") if isinstance(fn, dict) else entry.get("name")
            if name:
                names.append(str(name))
    return names


def _as_message(message: Any) -> dict:
    role = {
        "human": "user",
        "ai": "assistant",
        "system": "system",
        "tool": "tool",
    }.get(getattr(message, "type", ""), getattr(message, "type", "user"))
    return {"role": role, "content": getattr(message, "content", "")}


class LangGraphRecorder(BaseCallbackHandler):
    """Collect decisions from one or more LangGraph runs.

        recorder = LangGraphRecorder(session="thread-1")
        app.invoke(state, {"callbacks": [recorder]})
        refs = recorder.references()

    `session` groups steps into a trajectory. Pass a new one per conversation,
    or set `session_from` to read it off the run's metadata.
    """

    def __init__(
        self,
        session: str = "run",
        *,
        session_from: str | None = None,
        finish_action: str = FINISH,
        include_nodes: set[str] | None = None,
    ):
        self.session = session
        self.session_from = session_from
        self.finish_action = finish_action
        self.include_nodes = include_nodes
        self.steps: list[TraceStep] = []
        self.skipped_no_tools = 0
        self._pending: dict[Any, dict] = {}

    # -- LangChain callback protocol -------------------------------------

    def on_chat_model_start(self, serialized, messages, **kwargs) -> None:
        metadata = kwargs.get("metadata") or {}
        node = metadata.get("langgraph_node")
        if self.include_nodes is not None and node not in self.include_nodes:
            return
        tools = _tool_names(kwargs.get("invocation_params"))
        if not tools:
            self.skipped_no_tools += 1
            return
        batch = messages[0] if messages else []
        session = self.session
        if self.session_from:
            session = str(metadata.get(self.session_from, self.session))
        self._pending[kwargs.get("run_id")] = {
            "session": session,
            "role": str(node or "graph"),
            "messages": [_as_message(m) for m in batch],
            "tools": tools,
            "step": metadata.get("langgraph_step"),
        }

    def on_llm_end(self, response, **kwargs) -> None:
        started = self._pending.pop(kwargs.get("run_id"), None)
        if started is None:
            return
        chosen = self._chosen(response)
        options = list(started["tools"])
        if self.finish_action:
            options.append(self.finish_action)
        if chosen is None:
            chosen = self.finish_action
        if chosen not in options:
            # a tool the model named but was not offered; keep the step honest
            options.append(chosen)
        self.steps.append(
            TraceStep(
                session=started["session"],
                role=started["role"],
                messages=started["messages"],
                admissible=options,
                action=chosen,
                meta={"langgraph_step": started["step"]},
            )
        )

    def on_llm_error(self, error, **kwargs) -> None:
        self._pending.pop(kwargs.get("run_id"), None)

    # -- results ----------------------------------------------------------

    @staticmethod
    def _chosen(response) -> str | None:
        try:
            generation = response.generations[0][0]
        except (AttributeError, IndexError):
            return None
        calls = getattr(getattr(generation, "message", None), "tool_calls", None)
        if not calls:
            return None
        first = calls[0]
        return str(first.get("name") if isinstance(first, dict) else first)

    def references(self, **kwargs):
        from .trace import build_references

        kwargs.setdefault("generator", "langgraph")
        return build_references(self.steps, **kwargs)
