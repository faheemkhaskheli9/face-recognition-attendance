"""Tests for the enrollment pipeline and store."""
from __future__ import annotations

import json

import numpy as np
import pytest

from src.embeddings import HashEmbedder
from src.enrollment import (
    DuplicateEnrollmentError,
    EnrollmentError,
    EnrollmentStore,
    enroll_person,
)


def test_enroll_stores_encoding_and_persists(tmp_path, face_a, face_a_variant):
    store_path = tmp_path / "enr.json"
    store = EnrollmentStore(store_path)
    rec = enroll_person(store, "alice", "Alice Doe", [face_a, face_a_variant], HashEmbedder())

    assert rec.person_id == "alice"
    assert rec.num_samples == 2
    assert len(rec.embedding) == 128
    assert store_path.is_file()

    # Reload from disk -> same person present.
    reloaded = EnrollmentStore(store_path)
    assert "alice" in reloaded
    assert reloaded.get("alice").name == "Alice Doe"


def test_duplicate_enrollment_updates_by_default(tmp_path, face_a, face_b):
    store = EnrollmentStore(tmp_path / "enr.json")
    enroll_person(store, "alice", "Alice", [face_a], HashEmbedder())
    updated = enroll_person(store, "alice", "Alice Renamed", [face_b], HashEmbedder())
    assert len(store) == 1
    assert updated.name == "Alice Renamed"


def test_duplicate_enrollment_can_be_rejected(tmp_path, face_a, face_b):
    store = EnrollmentStore(tmp_path / "enr.json")
    enroll_person(store, "alice", "Alice", [face_a], HashEmbedder())
    with pytest.raises(DuplicateEnrollmentError):
        enroll_person(store, "alice", "Alice", [face_b], HashEmbedder(), on_duplicate="reject")


def test_same_name_different_ids_do_not_collide(tmp_path, face_a, face_b):
    store = EnrollmentStore(tmp_path / "enr.json")
    enroll_person(store, "jsmith1", "John Smith", [face_a], HashEmbedder())
    enroll_person(store, "jsmith2", "John Smith", [face_b], HashEmbedder())
    assert len(store) == 2


def test_blank_person_id_rejected(tmp_path, face_a):
    store = EnrollmentStore(tmp_path / "enr.json")
    with pytest.raises(EnrollmentError, match="person_id"):
        enroll_person(store, "  ", "Nobody", [face_a], HashEmbedder())


def test_no_images_rejected(tmp_path):
    store = EnrollmentStore(tmp_path / "enr.json")
    with pytest.raises(EnrollmentError, match="no images"):
        enroll_person(store, "alice", "Alice", [], HashEmbedder())


def test_flush_is_atomic_on_failure(tmp_path, face_a, monkeypatch):
    store_path = tmp_path / "enr.json"
    store = EnrollmentStore(store_path)
    import src.enrollment as enr

    monkeypatch.setattr(enr.os, "replace", lambda s, d: (_ for _ in ()).throw(OSError("boom")))
    with pytest.raises(OSError):
        enroll_person(store, "alice", "Alice", [face_a], HashEmbedder())

    assert not store_path.exists()
    assert not store_path.with_suffix(".json.tmp").exists()


def test_remove_person(tmp_path, face_a):
    store = EnrollmentStore(tmp_path / "enr.json")
    enroll_person(store, "alice", "Alice", [face_a], HashEmbedder())
    assert store.remove("alice") is True
    assert store.remove("alice") is False
