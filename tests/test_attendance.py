"""Tests for Phase 2 check-in logging and history storage."""
from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest

from src.attendance import AttendanceStore


def test_check_in_persists_and_reads_back(tmp_path):
    with AttendanceStore(tmp_path / "attendance.db") as store:
        result = store.check_in("alice")
        assert result.logged is True
        assert result.record.person_id == "alice"

        history = store.history("alice")
        assert len(history) == 1
        assert history[0].id == result.record.id
        assert history[0].timestamp == result.record.timestamp


def test_check_in_survives_reopening_the_store(tmp_path):
    db = tmp_path / "attendance.db"
    with AttendanceStore(db) as store:
        store.check_in("alice", timestamp=datetime(2026, 1, 1, tzinfo=timezone.utc))

    with AttendanceStore(db) as store:
        history = store.history("alice")
    assert len(history) == 1


def test_in_memory_store_works_for_tests():
    with AttendanceStore(":memory:") as store:
        store.check_in("alice")
        assert len(store.history()) == 1


def test_duplicate_checkin_within_window_is_deduplicated():
    with AttendanceStore(":memory:", dedup_window_seconds=60) as store:
        t0 = datetime(2026, 1, 1, 12, 0, 0, tzinfo=timezone.utc)
        first = store.check_in("alice", timestamp=t0)
        second = store.check_in("alice", timestamp=t0 + timedelta(seconds=30))

        assert first.logged is True
        assert second.logged is False
        assert second.record.id == first.record.id
        assert store.history("alice") == [first.record]


def test_checkin_outside_window_is_logged_separately():
    with AttendanceStore(":memory:", dedup_window_seconds=60) as store:
        t0 = datetime(2026, 1, 1, 12, 0, 0, tzinfo=timezone.utc)
        first = store.check_in("alice", timestamp=t0)
        second = store.check_in("alice", timestamp=t0 + timedelta(seconds=61))

        assert first.logged is True
        assert second.logged is True
        assert second.record.id != first.record.id
        assert len(store.history("alice")) == 2


def test_different_people_never_deduplicate_against_each_other():
    with AttendanceStore(":memory:", dedup_window_seconds=60) as store:
        t0 = datetime(2026, 1, 1, 12, 0, 0, tzinfo=timezone.utc)
        alice = store.check_in("alice", timestamp=t0)
        bob = store.check_in("bob", timestamp=t0)

        assert alice.logged is True
        assert bob.logged is True
        assert len(store.history()) == 2


def test_history_without_person_id_returns_everyone_oldest_first():
    with AttendanceStore(":memory:") as store:
        t0 = datetime(2026, 1, 1, 12, 0, 0, tzinfo=timezone.utc)
        store.check_in("bob", timestamp=t0)
        store.check_in("alice", timestamp=t0 + timedelta(hours=1))

        history = store.history()
        assert [r.person_id for r in history] == ["bob", "alice"]


def test_empty_person_id_is_rejected():
    with AttendanceStore(":memory:") as store:
        with pytest.raises(ValueError):
            store.check_in("   ")


def test_naive_timestamp_is_rejected():
    with AttendanceStore(":memory:") as store:
        with pytest.raises(ValueError):
            store.check_in("alice", timestamp=datetime(2026, 1, 1))


def test_negative_dedup_window_is_rejected():
    with pytest.raises(ValueError):
        AttendanceStore(":memory:", dedup_window_seconds=-1)
