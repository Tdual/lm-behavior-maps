"""The Fisher-Rao log map, for when alpha=0 coordinates have to be added.

At alpha=0 each reference step contributes sqrt(pi), which has unit norm, so
every agent sits on a product of unit spheres and the difference of two agents
is a chord -- not a tangent vector. Chords on a curved surface do not add.

The log map at a base point projects the chord onto the tangent plane and
stretches it to the arc length, which is the displacement that does add:

    Log_b(p) = (theta / sin theta) (sqrt p - BC sqrt b),  theta = arccos(BC)

applied per reference step, since the manifold is a product.

The correction is real but small: on measured data it changed the composition
cosine only in the third decimal. Curvature is not what costs composition its
accuracy -- interaction between the components is, and that costs alpha=1 just
as much.
"""

from __future__ import annotations

import numpy as np

from .types import PolicyVectors


def fisher_rao_log(vectors: PolicyVectors, base: PolicyVectors) -> np.ndarray:
    """Displacement of `vectors` from `base`, in the tangent space at `base`."""
    out = []
    m = vectors.n_steps
    for p, b in zip(vectors.policies, base.policies, strict=True):
        sp, sb = np.sqrt(p), np.sqrt(b)
        bc = float(np.clip(np.dot(sp, sb), -1.0, 1.0))
        theta = np.arccos(bc)
        s = np.sin(theta)
        out.append((sp - bc * sb) * (theta / s if s > 1e-12 else 1.0))
    return np.concatenate(out) / np.sqrt(m)


def step_angles(vectors: PolicyVectors, base: PolicyVectors) -> np.ndarray:
    """Per-step angle to the base point, in radians.

    How far around the sphere each displacement reaches, and therefore how far
    the flat approximation is being pushed. A maximum near pi/2 means chord and
    arc have visibly parted company and the log map is earning its keep.
    """
    out = np.empty(vectors.n_steps, dtype=np.float64)
    for k, (p, b) in enumerate(zip(vectors.policies, base.policies, strict=True)):
        bc = float(np.clip(np.dot(np.sqrt(p), np.sqrt(b)), -1.0, 1.0))
        out[k] = np.arccos(bc)
    return out
