"""Tests for the Phase 1 CLI."""
from __future__ import annotations

import numpy as np

from src.main import main


def _write_face(path, seed):
    rng = np.random.default_rng(seed)
    np.save(path, (rng.random((48, 48, 3)) * 255).astype(np.uint8))
    return str(path.with_suffix(".npy")) if path.suffix != ".npy" else str(path)


def test_enroll_list_remove_roundtrip(tmp_path, capsys):
    store = tmp_path / "enr.json"
    img1 = tmp_path / "a1.npy"
    img2 = tmp_path / "a2.npy"
    _write_face(img1, 1)
    _write_face(img2, 2)

    rc = main(
        ["--store", str(store), "enroll", "--person-id", "alice", "--name", "Alice",
         "--images", str(img1), str(img2)]
    )
    assert rc == 0
    assert "enrolled alice" in capsys.readouterr().out

    rc = main(["--store", str(store), "list"])
    assert rc == 0
    assert "alice" in capsys.readouterr().out

    rc = main(["--store", str(store), "remove", "--person-id", "alice"])
    assert rc == 0
    assert "removed alice" in capsys.readouterr().out


def test_recognize_matches_enrolled_person(tmp_path, capsys):
    store = tmp_path / "enr.json"
    img1 = tmp_path / "a1.npy"
    _write_face(img1, 1)

    rc = main(["--store", str(store), "enroll", "--person-id", "alice", "--images", str(img1)])
    assert rc == 0
    capsys.readouterr()

    rc = main(
        ["--store", str(store), "recognize", "--image", str(img1),
         "--attendance-db", str(tmp_path / "attendance.db")]
    )
    assert rc == 0
    assert "alice" in capsys.readouterr().out


def test_recognize_logs_a_checkin_by_default(tmp_path, capsys):
    store = tmp_path / "enr.json"
    img1 = tmp_path / "a1.npy"
    _write_face(img1, 1)
    db = tmp_path / "attendance.db"

    main(["--store", str(store), "enroll", "--person-id", "alice", "--images", str(img1)])
    capsys.readouterr()

    rc = main(["--store", str(store), "recognize", "--image", str(img1), "--attendance-db", str(db)])
    assert rc == 0
    out = capsys.readouterr().out
    assert "checked in alice" in out

    from src.attendance import AttendanceStore

    with AttendanceStore(db) as attendance:
        history = attendance.history("alice")
    assert len(history) == 1


def test_recognize_with_no_checkin_does_not_log(tmp_path, capsys):
    store = tmp_path / "enr.json"
    img1 = tmp_path / "a1.npy"
    _write_face(img1, 1)
    db = tmp_path / "attendance.db"

    main(["--store", str(store), "enroll", "--person-id", "alice", "--images", str(img1)])
    capsys.readouterr()

    rc = main(
        ["--store", str(store), "recognize", "--image", str(img1),
         "--attendance-db", str(db), "--no-checkin"]
    )
    assert rc == 0
    assert not db.exists()


def test_recognize_reports_unknown_for_unenrolled_face(tmp_path, capsys):
    store = tmp_path / "enr.json"
    img1 = tmp_path / "a1.npy"
    img2 = tmp_path / "a2.npy"
    _write_face(img1, 1)
    _write_face(img2, 2)

    rc = main(["--store", str(store), "enroll", "--person-id", "alice", "--images", str(img1)])
    assert rc == 0
    capsys.readouterr()

    rc = main(
        ["--store", str(store), "recognize", "--image", str(img2),
         "--attendance-db", str(tmp_path / "attendance.db")]
    )
    assert rc == 0
    assert "unknown" in capsys.readouterr().out


def test_recognize_against_empty_store_reports_error(tmp_path, capsys):
    img1 = tmp_path / "a1.npy"
    _write_face(img1, 1)

    rc = main(["--store", str(tmp_path / "empty.json"), "recognize", "--image", str(img1)])
    assert rc == 1
    assert "error" in capsys.readouterr().err


def test_enroll_missing_image_reports_error(tmp_path, capsys):
    rc = main(
        ["--store", str(tmp_path / "e.json"), "enroll", "--person-id", "x",
         "--images", str(tmp_path / "nope.npy")]
    )
    assert rc == 1
    assert "not found" in capsys.readouterr().err


def test_remove_unknown_person_returns_1(tmp_path, capsys):
    rc = main(["--store", str(tmp_path / "e.json"), "remove", "--person-id", "ghost"])
    assert rc == 1
    assert "not enrolled" in capsys.readouterr().err
