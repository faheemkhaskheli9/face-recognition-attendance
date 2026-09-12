"""Match a face embedding against enrolled people.

Returns the best-matching ``person_id`` only if its similarity to the query
clears a configurable threshold; otherwise returns ``UNKNOWN`` rather than the
nearest (but too-dissimilar) enrolled person -- a false match is worse than
admitting "don't know" for an attendance system.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from src.enrollment import EnrollmentRecord, EnrollmentStore

UNKNOWN = "unknown"

DEFAULT_THRESHOLD = 0.6


class NoEnrollmentsError(RuntimeError):
    """Raised when matching is attempted against an empty enrollment store."""


@dataclass(frozen=True)
class MatchResult:
    person_id: str
    """Matched person's id, or :data:`UNKNOWN` if nothing cleared the threshold."""
    similarity: float
    """Cosine similarity to the best candidate (0 enrollments aside, always set)."""
    name: str | None = None
    """Enrolled display name, or ``None`` when ``person_id`` is :data:`UNKNOWN`."""


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    denom = np.linalg.norm(a) * np.linalg.norm(b)
    if denom == 0:
        return 0.0
    return float(np.dot(a, b) / denom)


def match_embedding(
    embedding: np.ndarray,
    store: EnrollmentStore,
    *,
    threshold: float = DEFAULT_THRESHOLD,
) -> MatchResult:
    """Compare ``embedding`` against every record in ``store``.

    Returns the highest-similarity enrolled person if ``similarity >=
    threshold``, otherwise a result with ``person_id == UNKNOWN``. Raises
    :class:`NoEnrollmentsError` if the store has no enrolled people at all,
    since "no match" and "nothing to match against" are different failure
    modes a caller should be able to distinguish.
    """
    if not 0.0 <= threshold <= 1.0:
        raise ValueError(f"threshold must be in [0, 1], got {threshold}")

    records: list[EnrollmentRecord] = store.all()
    if not records:
        raise NoEnrollmentsError("no enrolled people to match against")

    best_record = max(records, key=lambda r: cosine_similarity(embedding, r.vector()))
    best_similarity = cosine_similarity(embedding, best_record.vector())

    if best_similarity >= threshold:
        return MatchResult(person_id=best_record.person_id, similarity=best_similarity, name=best_record.name)
    return MatchResult(person_id=UNKNOWN, similarity=best_similarity, name=None)
