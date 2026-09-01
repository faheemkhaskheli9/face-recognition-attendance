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
