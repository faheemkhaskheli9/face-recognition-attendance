"""Face enrollment: turn one or more images of a person into a stored encoding.

Storage model: a single JSON file mapping ``person_id -> EnrollmentRecord``.
``person_id`` is the unique key (robustness rule 6 -- names may collide, ids
must not). Writes go through a temp file + ``os.replace`` so a crash never
leaves a half-written registry (robustness rule 1 & 2).

Duplicate policy (documented choice): re-enrolling an existing ``person_id``
**updates** the record by default; pass ``on_duplicate="reject"`` to raise
instead.
"""
from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable, Literal

import numpy as np

from src.embeddings import FaceEmbedder, NoFaceFoundError

OnDuplicate = Literal["update", "reject"]


class EnrollmentError(RuntimeError):
    """Base class for enrollment failures."""


class DuplicateEnrollmentError(EnrollmentError):
    """Raised when re-enrolling an existing person_id with on_duplicate='reject'."""


@dataclass
class EnrollmentRecord:
    person_id: str
    name: str
    embedding: list[float]
    num_samples: int
    backend: str
    updated_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())

    def vector(self) -> np.ndarray:
        return np.asarray(self.embedding, dtype=np.float32)


class EnrollmentStore:
    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self._records: dict[str, EnrollmentRecord] = {}
        if self.path.is_file():
            self._load()

    def _load(self) -> None:
        data = json.loads(self.path.read_text(encoding="utf-8"))
        self._records = {
            pid: EnrollmentRecord(**rec) for pid, rec in data.get("people", {}).items()
        }

    def _flush(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "version": 1,
            "people": {pid: asdict(rec) for pid, rec in self._records.items()},
        }
        tmp = self.path.with_suffix(self.path.suffix + ".tmp")
        try:
            tmp.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
            os.replace(tmp, self.path)
        except BaseException:
            tmp.unlink(missing_ok=True)
            raise

    def __contains__(self, person_id: str) -> bool:
        return person_id in self._records

    def __len__(self) -> int:
        return len(self._records)

    def get(self, person_id: str) -> EnrollmentRecord | None:
        return self._records.get(person_id)

    def all(self) -> list[EnrollmentRecord]:
        return sorted(self._records.values(), key=lambda r: r.person_id)

    def remove(self, person_id: str) -> bool:
        existed = self._records.pop(person_id, None) is not None
        if existed:
            self._flush()
        return existed

    def enroll(
        self,
        person_id: str,
        name: str,
        images: Iterable[np.ndarray],
        embedder: FaceEmbedder,
        *,
        on_duplicate: OnDuplicate = "update",
    ) -> EnrollmentRecord:
        person_id = (person_id or "").strip()
        if not person_id:
            raise EnrollmentError("person_id must be a non-empty string")
        if person_id in self._records and on_duplicate == "reject":
            raise DuplicateEnrollmentError(f"person_id {person_id!r} is already enrolled")

        vectors: list[np.ndarray] = []
        for i, image in enumerate(images):
            try:
                vectors.append(embedder.embed(image))
            except NoFaceFoundError as exc:
                raise EnrollmentError(f"image {i}: {exc}") from exc
        if not vectors:
            raise EnrollmentError("no images supplied for enrollment")

        mean = np.mean(np.stack(vectors), axis=0)
        norm = float(np.linalg.norm(mean))
        if norm:
            mean = mean / norm

        record = EnrollmentRecord(
            person_id=person_id,
            name=name.strip() or person_id,
            embedding=[float(x) for x in mean],
            num_samples=len(vectors),
            backend=embedder.name,
        )
        self._records[person_id] = record
        self._flush()
        return record


def enroll_person(
    store: EnrollmentStore,
    person_id: str,
    name: str,
    images: Iterable[np.ndarray],
    embedder: FaceEmbedder,
    *,
    on_duplicate: OnDuplicate = "update",
) -> EnrollmentRecord:
    """Module-level convenience wrapper around :meth:`EnrollmentStore.enroll`."""
    return store.enroll(person_id, name, images, embedder, on_duplicate=on_duplicate)
