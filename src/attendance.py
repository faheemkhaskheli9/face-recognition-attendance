"""Check-in logging and attendance history storage (SQLite).

Phase 2 scope: log a check-in event (``person_id``, timestamp) each time
:func:`src.matching.match_embedding` finds a known person, and let that
history be queried back.

Duplicate policy (documented choice, per the issue's acceptance criteria): a
check-in for the same ``person_id`` within :data:`DEFAULT_DEDUP_WINDOW_SECONDS`
of their most recent logged check-in is treated as a duplicate -- e.g.
someone standing in front of the camera for a few seconds fires several
recognitions, which should count as one attendance event, not several. The
duplicate call returns the existing row unchanged rather than silently
dropping the request.
"""
from __future__ import annotations

import sqlite3
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path

DEFAULT_DEDUP_WINDOW_SECONDS = 60.0


@dataclass(frozen=True)
class CheckInRecord:
    id: int
    person_id: str
    timestamp: str  # ISO-8601, always timezone-aware (UTC unless caller passes otherwise)


@dataclass(frozen=True)
class CheckInResult:
    record: CheckInRecord
    """The check-in as it stands after this call: the new row if ``logged``,
    otherwise the existing recent row this call deduplicated against."""
    logged: bool
    """True if a new row was written; False if deduplicated."""


class AttendanceStore:
    """SQLite-backed check-in log. ``path=":memory:"`` is supported for tests."""

    def __init__(
        self, path: str | Path, *, dedup_window_seconds: float = DEFAULT_DEDUP_WINDOW_SECONDS
    ) -> None:
        if dedup_window_seconds < 0:
            raise ValueError(f"dedup_window_seconds must be >= 0, got {dedup_window_seconds}")
        self.path = str(path)
        if self.path != ":memory:":
            Path(self.path).parent.mkdir(parents=True, exist_ok=True)
        self.dedup_window_seconds = dedup_window_seconds
        self._conn = sqlite3.connect(self.path)
        self._conn.execute(
            """
            CREATE TABLE IF NOT EXISTS checkins (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                person_id TEXT NOT NULL,
                timestamp TEXT NOT NULL
            )
            """
        )
        self._conn.execute("CREATE INDEX IF NOT EXISTS idx_checkins_person_id ON checkins(person_id)")
        self._conn.commit()

    def close(self) -> None:
        self._conn.close()

    def __enter__(self) -> "AttendanceStore":
        return self

    def __exit__(self, *exc_info: object) -> None:
        self.close()

    def check_in(self, person_id: str, *, timestamp: datetime | None = None) -> CheckInResult:
        """Log a check-in for ``person_id`` at ``timestamp`` (default: now, UTC).

        Returns whether a new row was written or an existing recent one was
        reused under the dedup policy. Raises :class:`ValueError` for an
        empty ``person_id`` or a naive (no-tzinfo) ``timestamp`` -- silently
        treating naive time as UTC would risk comparing incompatible clocks.
        """
        person_id = (person_id or "").strip()
        if not person_id:
            raise ValueError("person_id must be a non-empty string")
        now = timestamp if timestamp is not None else datetime.now(timezone.utc)
        if now.tzinfo is None:
            raise ValueError("timestamp must be timezone-aware")

        last = self._last_checkin(person_id)
        if last is not None:
            last_dt = datetime.fromisoformat(last.timestamp)
            if timedelta(0) <= now - last_dt < timedelta(seconds=self.dedup_window_seconds):
                return CheckInResult(record=last, logged=False)

        cursor = self._conn.execute(
            "INSERT INTO checkins (person_id, timestamp) VALUES (?, ?)",
            (person_id, now.isoformat()),
        )
        self._conn.commit()
        record = CheckInRecord(id=cursor.lastrowid, person_id=person_id, timestamp=now.isoformat())
        return CheckInResult(record=record, logged=True)

    def _last_checkin(self, person_id: str) -> CheckInRecord | None:
        row = self._conn.execute(
            "SELECT id, person_id, timestamp FROM checkins WHERE person_id = ? "
            "ORDER BY timestamp DESC, id DESC LIMIT 1",
            (person_id,),
        ).fetchone()
        return CheckInRecord(*row) if row is not None else None

    def history(self, person_id: str | None = None) -> list[CheckInRecord]:
        """All check-ins, oldest first, optionally filtered to one ``person_id``."""
        if person_id is None:
            rows = self._conn.execute(
                "SELECT id, person_id, timestamp FROM checkins ORDER BY timestamp ASC, id ASC"
            ).fetchall()
        else:
            rows = self._conn.execute(
                "SELECT id, person_id, timestamp FROM checkins WHERE person_id = ? "
                "ORDER BY timestamp ASC, id ASC",
                (person_id,),
            ).fetchall()
        return [CheckInRecord(*row) for row in rows]
