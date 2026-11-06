"""Build a tiny synthetic reference set, so the quickstart needs no benchmark.

Real reference trajectories come from somewhere else: logs of a running system,
an environment with an admissible-action list, or a scripted oracle. Whatever
produces them is part of the measurement -- record it in `generator`.
"""

import random

import lm_behavior as lb

ROOMS = ["kitchen", "bedroom", "bathroom", "living room"]
OBJECTS = ["mug", "apple", "book", "soap", "knife"]
RECEPTACLES = ["fridge", "drawer", "shelf", "sink", "countertop"]


def build(n_trajectories: int = 6, n_steps: int = 8, seed: int = 0) -> lb.ReferenceSet:
    rng = random.Random(seed)
    trajectories = []
    for _ in range(n_trajectories):
        obj, dest = rng.choice(OBJECTS), rng.choice(RECEPTACLES)
        task = f"put the {obj} in the {dest}"
        room = rng.choice(ROOMS)
        steps = []
        for t in range(n_steps):
            admissible = (
                [f"go to {r}" for r in ROOMS]
                + [f"take {obj}", f"open {dest}", f"put {obj} in {dest}", "look", "inventory"]
            )
            steps.append(
                lb.Step(
                    observation=f"You are in the {room}. You see a {dest} and a {obj}."
                    if t == 0
                    else f"Step {t}. You are in the {room}.",
                    admissible=admissible,
                    action=rng.choice(admissible),
                )
            )
            room = rng.choice(ROOMS)
        trajectories.append(
            lb.Trajectory(
                task=task,
                initial_observation=f"You are in the {rng.choice(ROOMS)}.",
                steps=steps,
            )
        )
    return lb.ReferenceSet(
        trajectories=trajectories, generator=f"examples/make_refs.py seed={seed}"
    )


if __name__ == "__main__":
    refs = build()
    lb.save_references(refs, "examples/refs.json")
    print(f"wrote examples/refs.json: {refs.n_trajectories} trajectories, {refs.n_steps} steps")
