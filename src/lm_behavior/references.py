"""Read and write reference sets as JSON.

Keep the file with the map it produced. A map is only comparable with another
map built on the same reference set, so the reference set is part of the
result, not scratch input.
"""

from __future__ import annotations

import json
from pathlib import Path

from .types import ReferenceSet, Step, Trajectory


def to_dict(refs: ReferenceSet) -> dict:
    return {
        "generator": refs.generator,
        "trajectories": [
            {
                "task": t.task,
                "initial_observation": t.initial_observation,
                "steps": [
                    {
                        "observation": s.observation,
                        "admissible": s.admissible,
                        "action": s.action,
                        **({"context": s.context} if s.context is not None else {}),
                        **({"role": s.role} if s.role is not None else {}),
                    }
                    for s in t.steps
                ],
            }
            for t in refs.trajectories
        ],
    }


def from_dict(data: dict) -> ReferenceSet:
    return ReferenceSet(
        generator=data.get("generator"),
        trajectories=[
            Trajectory(
                task=t["task"],
                initial_observation=t["initial_observation"],
                steps=[
                    Step(
                        observation=s["observation"],
                        admissible=list(s["admissible"]),
                        action=s["action"],
                        context=s.get("context"),
                        role=s.get("role"),
                    )
                    for s in t["steps"]
                ],
            )
            for t in data["trajectories"]
        ],
    )


def save(refs: ReferenceSet, path: str | Path) -> None:
    Path(path).write_text(
        json.dumps(to_dict(refs), ensure_ascii=False, indent=1), encoding="utf-8"
    )


def load(path: str | Path) -> ReferenceSet:
    return from_dict(json.loads(Path(path).read_text(encoding="utf-8")))
