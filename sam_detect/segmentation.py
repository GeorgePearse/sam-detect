"""Segmentation backends for sam-detect.

This module intentionally keeps the interface lightweight so it can be
swapped with a real SAM2 + TensorRT implementation later on.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable, Protocol, Sequence


@dataclass
class SegmentationResult:
    """Container describing a detected instance."""

    mask: Sequence[Sequence[bool]]
    bbox: tuple[int, int, int, int]


class SegmentationBackend(Protocol):
    """Callable interface for segmentation backends."""

    def __call__(self, image: "ImageType") -> Iterable[SegmentationResult]: ...


ImageType = Any  # Placeholder alias until real image types are available


class NaiveSegmenter:
    """Fallback segmenter that emits a single full-frame mask.

    This is helpful for smoke testing the rest of the pipeline when SAM2
    acceleration is not available.
    """

    def __call__(self, image: ImageType) -> Iterable[SegmentationResult]:
        width, height = _infer_dimensions(image)
        mask = [[True for _ in range(width)] for _ in range(height)]
        yield SegmentationResult(mask=mask, bbox=(0, 0, width, height))


def _infer_dimensions(image: ImageType) -> tuple[int, int]:
    """Best-effort inference of image dimensions.

    Accepts common Pillow-like objects or 2D/3D array-likes. Falls back to a
    1x1 canvas to keep the pipeline moving.
    """

    if hasattr(image, "size") and isinstance(image.size, tuple):
        # Pillow Image uses (width, height)
        width, height = image.size[:2]
        return int(width), int(height)

    if hasattr(image, "shape"):
        shape = getattr(image, "shape")
        if len(shape) >= 2:
            height, width = shape[0], shape[1]
            return int(width), int(height)

    return 1, 1
