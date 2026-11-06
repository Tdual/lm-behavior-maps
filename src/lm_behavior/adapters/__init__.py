"""Turn a framework's execution trace into reference trajectories."""

from .trace import FINISH, ChatHarness, TraceStep, build_references, render

__all__ = [
    "FINISH",
    "ChatHarness",
    "CrewAIRecorder",
    "LangGraphRecorder",
    "TraceStep",
    "build_references",
    "render",
]


def __getattr__(name):  # keep langgraph and crewai out of the import path
    if name == "LangGraphRecorder":
        from .langgraph import LangGraphRecorder

        return LangGraphRecorder
    if name == "CrewAIRecorder":
        from .crewai import CrewAIRecorder

        return CrewAIRecorder
    raise AttributeError(name)
