"""The identities the alpha=0 coordinate rests on, checked numerically.

These are the unit checks the reproduction used before trusting any number:
the isometry, the local relation to KL, and the direction of the global bound.
"""

import numpy as np
import pytest

import lm_behavior as lb
from lm_behavior.kl import per_step_kl


def random_simplex(rng, n):
    p = rng.dirichlet(np.ones(n) * rng.uniform(0.3, 3.0))
    return p


def test_hellinger_identity():
    """||sqrt p - sqrt q||^2 == 2(1 - BC)."""
    rng = np.random.default_rng(0)
    for _ in range(500):
        n = int(rng.integers(2, 12))
        p, q = random_simplex(rng, n), random_simplex(rng, n)
        chord = float(((np.sqrt(p) - np.sqrt(q)) ** 2).sum())
        bc = float(np.sqrt(p * q).sum())
        assert chord == pytest.approx(2 * (1 - bc), abs=1e-12)


def test_global_lower_bound():
    """||sqrt p - sqrt q||^2 <= KL(p||q). Never an overestimate."""
    rng = np.random.default_rng(1)
    for _ in range(5000):
        n = int(rng.integers(2, 10))
        p, q = random_simplex(rng, n), random_simplex(rng, n)
        chord = float(((np.sqrt(p) - np.sqrt(q)) ** 2).sum())
        assert chord <= per_step_kl(p, q) + 1e-9


def test_local_relation_to_kl():
    """KL -> 2||sqrt p - sqrt q||^2 as the two distributions close up."""
    rng = np.random.default_rng(2)
    ratios = []
    for _ in range(200):
        n = int(rng.integers(3, 8))
        p = random_simplex(rng, n)
        q = p + rng.normal(scale=1e-4, size=n)
        q = np.clip(q, 1e-9, None)
        q /= q.sum()
        chord = float(((np.sqrt(p) - np.sqrt(q)) ** 2).sum())
        sym = per_step_kl(p, q) + per_step_kl(q, p)
        if chord > 1e-14:
            ratios.append(sym / (4 * chord))
    assert np.median(ratios) == pytest.approx(1.0, abs=1e-3)


def test_chord_and_arc_agree_at_small_angle():
    rng = np.random.default_rng(3)
    p = random_simplex(rng, 6)
    q = np.clip(p + rng.normal(scale=1e-5, size=6), 1e-12, None)
    q /= q.sum()
    bc = np.clip(float(np.sqrt(p * q).sum()), 0, 1)
    chord = np.sqrt(2 * (1 - bc))
    arc = np.arccos(bc)
    assert chord == pytest.approx(arc, rel=1e-4)


def test_alpha0_coordinate_is_on_the_unit_sphere(vectors):
    for v in vectors:
        c = lb.embed(v, alpha=0.0)
        assert float(np.linalg.norm(c.z)) == pytest.approx(1.0, abs=1e-12)


def test_bhattacharyya_matches_direct_computation(vectors):
    a, b = lb.embed(vectors[0], alpha=0.0), lb.embed(vectors[1], alpha=0.0)
    got = lb.bhattacharyya(a, b)
    pairs = zip(vectors[0].policies, vectors[1].policies, strict=True)
    want = np.array([float(np.sqrt(p * q).sum()) for p, q in pairs])
    assert np.allclose(got, want, atol=1e-12)


def test_fisher_rao_is_at_least_the_chord(vectors):
    a, b = lb.embed(vectors[0], alpha=0.0), lb.embed(vectors[1], alpha=0.0)
    chord = lb.distance(a, b, "hellinger")
    arc = lb.distance(a, b, "fisher_rao")
    assert arc >= chord - 1e-12
    assert chord / arc > 0.9  # the 2-4% gap seen in measurement, not a blow-up
