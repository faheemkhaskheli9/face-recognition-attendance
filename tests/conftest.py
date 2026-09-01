"""Shared fixtures: synthetic 'face' images for offline tests."""
from __future__ import annotations

import numpy as np
import pytest


def _synthetic_face(seed: int, shape=(64, 64, 3)) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return (rng.random(shape) * 255).astype(np.uint8)


@pytest.fixture
def face_a() -> np.ndarray:
    return _synthetic_face(1)


@pytest.fixture
def face_a_variant() -> np.ndarray:
    # Same seed base, small perturbation -> a "second photo of person A".
    base = _synthetic_face(1).astype(np.int16)
    noise = np.random.default_rng(99).integers(-4, 5, size=base.shape)
    return np.clip(base + noise, 0, 255).astype(np.uint8)


@pytest.fixture
def face_b() -> np.ndarray:
    return _synthetic_face(2)
