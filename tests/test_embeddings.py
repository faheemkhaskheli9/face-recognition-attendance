"""Tests for the face embedding backends."""
from __future__ import annotations

import numpy as np
import pytest

from src.embeddings import (
    EMBEDDING_DIM,
    HashEmbedder,
    NoFaceFoundError,
    load_embedder,
)


def test_hash_embedder_is_deterministic_and_unit_norm(face_a):
    emb = HashEmbedder()
    v1 = emb.embed(face_a)
    v2 = emb.embed(face_a.copy())
    assert v1.shape == (EMBEDDING_DIM,)
    assert np.allclose(v1, v2)
    assert np.isclose(np.linalg.norm(v1), 1.0, atol=1e-5)


def test_hash_embedder_distinguishes_faces(face_a, face_b):
    emb = HashEmbedder()
    assert not np.allclose(emb.embed(face_a), emb.embed(face_b))


def test_hash_embedder_rejects_empty_image():
    with pytest.raises(NoFaceFoundError):
        HashEmbedder().embed(np.empty((0, 0, 3), dtype=np.uint8))


def test_load_embedder_unknown_backend():
    with pytest.raises(ValueError, match="unknown embedder"):
        load_embedder("magic")


def test_load_embedder_default_is_hash():
    assert load_embedder().name == "hash"
