"""Cosine ranking in the shortlist: clipped, and safe on an empty document matrix.

Ported from upstream v0.3.20 ``tests/test_shortlist_cosine.py`` (issue #158). Float rounding can
push the cosine of two nearly parallel vectors just above 1.0, and ``np.linalg.norm`` raises on a
1-D empty document matrix before the zero-query branch can return. Both are guarded in ``_cosine``.
"""

import numpy as np

from laya_mlx.shortlist import _cosine

# Nearly parallel vectors whose *unclipped* cosine rounds above 1.0.
_NEAR_Q = np.array([669335.6983561065, 421969.494778055, 250162.44798431583], dtype=np.float64)
_NEAR_D = np.array([669335.6983561061, 421969.4947780549, 250162.44798431598], dtype=np.float64)


def test_cosine_identical_vectors():
    v = np.array([1.0, 2.0, 3.0], dtype=np.float64)
    docs = np.array([[1.0, 2.0, 3.0], [-1.0, -2.0, -3.0]], dtype=np.float64)
    sims = _cosine(v, docs)
    assert sims[0] == 1.0
    assert sims[1] == -1.0


def test_cosine_zero_vectors():
    v = np.zeros(4, dtype=np.float64)
    docs = np.ones((3, 4), dtype=np.float64)
    sims = _cosine(v, docs)
    assert np.all(sims == 0.0)


def test_cosine_empty_docs():
    v = np.ones(4, dtype=np.float64)
    docs = np.zeros((0, 4), dtype=np.float64)
    sims = _cosine(v, docs)
    assert len(sims) == 0


def test_cosine_similarity_is_clipped_to_one():
    raw = float(np.dot(_NEAR_D, _NEAR_Q) / (np.linalg.norm(_NEAR_D) * np.linalg.norm(_NEAR_Q)))
    assert raw > 1.0  # the guard is exercised, not vacuous
    sims = _cosine(_NEAR_Q, _NEAR_D.reshape(1, -1))
    assert sims[0] == 1.0


def test_cosine_is_never_out_of_range():
    rng = np.random.default_rng(0)
    for _ in range(200):
        dim = int(rng.integers(1, 8))
        q = rng.normal(size=dim)
        docs = rng.normal(size=(5, dim))
        sims = _cosine(q, docs)
        assert np.all(sims >= -1.0) and np.all(sims <= 1.0)
