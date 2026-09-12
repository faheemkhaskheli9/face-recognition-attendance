"""Tests for src.matching."""
from __future__ import annotations

import numpy as np
import pytest

from src.embeddings import HashEmbedder
from src.enrollment import EnrollmentStore
from src.matching import (
    NoEnrollmentsError,
    UNKNOWN,
    cosine_similarity,
    match_embedding,
)


@pytest.fixture
def store(tmp_path, face_a, face_b):
    embedder = HashEmbedder()
    s = EnrollmentStore(tmp_path / "enrollments.json")
    s.enroll("alice", "Alice", [face_a], embedder)
    s.enroll("bob", "Bob", [face_b], embedder)
    return s


def test_match_returns_person_above_threshold(store, face_a):
    embedder = HashEmbedder()
    # Re-embedding the exact enrolled image is a near-perfect match.
    query = embedder.embed(face_a)

    result = match_embedding(query, store, threshold=0.6)

    assert result.person_id == "alice"
    assert result.name == "Alice"
    assert result.similarity >= 0.6


def test_match_returns_unknown_below_threshold(store):
    # A random, unenrolled embedding should not match anyone.
    rng = np.random.default_rng(0)
    query = rng.random(128).astype(np.float32) - 0.5

    result = match_embedding(query, store, threshold=0.6)

    assert result.person_id == UNKNOWN
    assert result.name is None
    assert result.similarity < 0.6


def test_match_raises_on_empty_store(tmp_path):
    empty_store = EnrollmentStore(tmp_path / "empty.json")
    with pytest.raises(NoEnrollmentsError):
        match_embedding(np.zeros(128, dtype=np.float32), empty_store)


def test_match_rejects_invalid_threshold(store):
    with pytest.raises(ValueError):
        match_embedding(np.zeros(128, dtype=np.float32), store, threshold=1.5)


def test_cosine_similarity_identical_vectors_is_one():
    v = np.array([1.0, 2.0, 3.0])
    assert cosine_similarity(v, v) == pytest.approx(1.0)


def test_cosine_similarity_orthogonal_vectors_is_zero():
    a = np.array([1.0, 0.0])
    b = np.array([0.0, 1.0])
    assert cosine_similarity(a, b) == pytest.approx(0.0)


def test_cosine_similarity_zero_vector_is_zero_not_nan():
    a = np.zeros(4)
    b = np.array([1.0, 2.0, 3.0, 4.0])
    assert cosine_similarity(a, b) == 0.0
