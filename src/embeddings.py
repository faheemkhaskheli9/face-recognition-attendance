"""Face embedding backends.

The enrollment / recognition pipeline depends only on the :class:`FaceEmbedder`
protocol. Two implementations ship:

* :class:`DlibFaceEmbedder` -- the real backend, a thin wrapper over the
  ``face_recognition`` (dlib) library. It is imported lazily so the package,
  its tests, and CI do not require the dlib toolchain.
* :class:`HashEmbedder` -- a dependency-free, deterministic embedder used for
  offline runs and tests. It is **not** a real face model; it maps image pixels
  to a stable vector so the storage/matching logic can be exercised.
"""
from __future__ import annotations

import hashlib
from typing import Protocol, runtime_checkable

import numpy as np

EMBEDDING_DIM = 128


class NoFaceFoundError(RuntimeError):
    """Raised when an embedder cannot locate a face in an image."""


class MultipleFacesError(RuntimeError):
    """Raised when an image contains more than one face and one is required."""


@runtime_checkable
class FaceEmbedder(Protocol):
    name: str
    dim: int

    def embed(self, image: np.ndarray) -> np.ndarray:
        """Return a 1-D float32 embedding for the single face in ``image``."""


class HashEmbedder:
    """Deterministic stand-in embedder (no ML, no third-party deps)."""

    name = "hash"

    def __init__(self, dim: int = EMBEDDING_DIM) -> None:
        self.dim = dim

    def embed(self, image: np.ndarray) -> np.ndarray:
        arr = np.ascontiguousarray(image)
        if arr.size == 0:
            raise NoFaceFoundError("empty image")
        base = hashlib.sha256(arr.tobytes())
        base.update(np.asarray(arr.shape, dtype=np.int64).tobytes())
        seed = base.digest()
        need = self.dim * 4  # 4 bytes per uint32
        buf = bytearray()
        counter = 0
        while len(buf) < need:
            buf += hashlib.sha256(seed + counter.to_bytes(4, "big")).digest()
            counter += 1
        raw = np.frombuffer(bytes(buf[:need]), dtype=np.uint32).astype(np.float64)
        vec = (raw / np.iinfo(np.uint32).max) - 0.5
        norm = np.linalg.norm(vec)
        return (vec / norm).astype(np.float32) if norm else vec.astype(np.float32)


class DlibFaceEmbedder:
    """Real backend: face detection + 128-d encoding via ``face_recognition``."""

    name = "dlib"
    dim = EMBEDDING_DIM

    def __init__(self, *, model: str = "hog", require_single_face: bool = True) -> None:
        self.model = model
        self.require_single_face = require_single_face

    def embed(self, image: np.ndarray) -> np.ndarray:  # pragma: no cover - needs dlib
        try:
            import face_recognition
        except ImportError as exc:
            raise ImportError(
                "DlibFaceEmbedder requires the 'face_recognition' package "
                "(install the project's 'recognition' extra)"
            ) from exc

        boxes = face_recognition.face_locations(image, model=self.model)
        if not boxes:
            raise NoFaceFoundError("no face detected in image")
        if len(boxes) > 1 and self.require_single_face:
            raise MultipleFacesError(f"expected one face, found {len(boxes)}")
        encodings = face_recognition.face_encodings(image, known_face_locations=boxes[:1])
        if not encodings:
            raise NoFaceFoundError("face detected but encoding failed")
        return np.asarray(encodings[0], dtype=np.float32)


def load_embedder(name: str = "hash") -> FaceEmbedder:
    name = name.lower()
    if name == "hash":
        return HashEmbedder()
    if name in {"dlib", "face_recognition"}:
        return DlibFaceEmbedder()
    raise ValueError(f"unknown embedder backend: {name!r} (expected 'hash' or 'dlib')")
