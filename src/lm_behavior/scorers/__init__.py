"""Backends that can teacher-force a supplied action."""

from .function import FunctionScorer

__all__ = ["FunctionScorer", "HFScorer", "LMEvalScorer"]


def __getattr__(name):  # keep torch and lm-eval out of the import path
    if name == "HFScorer":
        from .hf import HFScorer

        return HFScorer
    if name == "LMEvalScorer":
        from .lmeval import LMEvalScorer

        return LMEvalScorer
    raise AttributeError(name)
