"""Image loading helpers.

Supports ``.npy`` arrays directly (used by tests and for offline runs) and
common image files via Pillow when it is installed.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np


class ImageLoadError(RuntimeError):
    """Raised when an image path cannot be read into an array."""


def load_image(path: str | Path) -> np.ndarray:
    p = Path(path)
    if not p.is_file():
        raise ImageLoadError(f"image file not found: {p}")

    if p.suffix.lower() == ".npy":
        arr = np.load(p)
    else:
        try:
            from PIL import Image
        except ImportError as exc:  # pragma: no cover - Pillow usually present
            raise ImageLoadError(
                f"reading {p.suffix} images requires Pillow (`pip install pillow`)"
            ) from exc
        with Image.open(p) as im:
            arr = np.asarray(im.convert("RGB"))

    if arr.ndim not in (2, 3) or arr.size == 0:
        raise ImageLoadError(f"unexpected image shape {arr.shape} for {p}")
    return arr
