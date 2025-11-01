"""Fading strategies for context-aware crops."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol, TYPE_CHECKING

from .segmentation import SegmentationResult

ImageType = Any

try:  # pragma: no cover - optional dependency
    import numpy as _np
except Exception:  # pragma: no cover - numpy not available
    _np = None  # type: ignore[assignment]

try:  # pragma: no cover - optional dependency
    from PIL import Image as _PILImage
except Exception:  # pragma: no cover - pillow not available
    _PILImage = None  # type: ignore[assignment]

if TYPE_CHECKING:  # pragma: no cover - type checking only
    import numpy as np  # noqa: F401


class FadeStrategy(Protocol):
    """Interface for applying a fading strategy to an image crop."""

    def apply(self, image: ImageType, result: SegmentationResult) -> ImageType: ...


@dataclass
class GaussianFade:
    """NumPy-backed Gaussian fade with graceful fallbacks."""

    sigma: float = 30.0
    min_fade: float = 0.05

    def apply(self, image: ImageType, result: SegmentationResult) -> ImageType:
        if _np is None:
            return image

        converted = _convert_to_numpy(image)
        if converted is None:
            return image

        array, dtype, mode, is_pil = converted
        mask = _np.asarray(result.mask, dtype=_np.bool_)

        if mask.ndim != 2:
            raise ValueError("Segmentation masks must be 2D")
        if array.ndim < 2:
            raise ValueError("Images must have at least two dimensions")

        height, width = mask.shape
        image_height, image_width = array.shape[0], array.shape[1]
        if (height, width) != (image_height, image_width):
            raise ValueError(
                "Mask shape does not match image dimensions: "
                f"mask={(height, width)} image={(image_height, image_width)}"
            )

        centroid_y, centroid_x = _mask_centroid(mask, image_width, image_height)
        weights = _compute_weights(
            mask, centroid_y, centroid_x, self.sigma, self.min_fade
        )
        faded = _apply_weights(array, weights, mask)

        if is_pil and _PILImage is not None:
            faded = _restore_dtype(faded, dtype)
            return _PILImage.fromarray(faded, mode=mode)

        return _restore_dtype(faded, dtype)


class IdentityFade:
    """No-op fade useful for unit tests and initial wiring."""

    def apply(
        self, image: ImageType, result: SegmentationResult
    ) -> ImageType:  # noqa: D401
        return image


def _convert_to_numpy(
    image: ImageType,
) -> tuple["np.ndarray", "np.dtype[Any]", str | None, bool] | None:
    if _np is None:
        return None

    if _PILImage is not None and isinstance(image, _PILImage.Image):
        array = _np.asarray(image)
        dtype = array.dtype
        return array.astype(_np.float32), dtype, image.mode, True

    if hasattr(image, "__array__"):
        array = _np.asarray(image)
        if array.ndim < 2:
            return None
        dtype = array.dtype
        return array.astype(_np.float32, copy=False), dtype, None, False

    return None


def _mask_centroid(
    mask: "np.ndarray",
    image_width: int,
    image_height: int,
) -> tuple[float, float]:
    assert _np is not None  # for type checkers

    indices = _np.argwhere(mask)
    if indices.size == 0:
        return (image_height - 1) / 2.0, (image_width - 1) / 2.0

    mean_y, mean_x = indices.mean(axis=0)
    return float(mean_y), float(mean_x)


def _compute_weights(
    mask: "np.ndarray",
    centroid_y: float,
    centroid_x: float,
    sigma: float,
    min_fade: float,
) -> "np.ndarray":
    assert _np is not None

    sigma = max(float(sigma), 1e-6)
    clamped_min = float(min(max(min_fade, 0.0), 1.0))

    grid_y, grid_x = _np.indices(mask.shape, dtype=_np.float32)
    distances_sq = (grid_y - centroid_y) ** 2 + (grid_x - centroid_x) ** 2
    weights = _np.exp(-distances_sq / (2.0 * sigma**2))
    weights = _np.clip(weights, clamped_min, 1.0)
    weights = _np.where(mask, 1.0, weights)
    return weights.astype(_np.float32, copy=False)


def _apply_weights(
    array: "np.ndarray",
    weights: "np.ndarray",
    mask: "np.ndarray",
) -> "np.ndarray":
    assert _np is not None

    if array.ndim == 2:
        return array * weights

    expanded = weights[..., _np.newaxis]
    return array * expanded


def _restore_dtype(array: "np.ndarray", dtype: "np.dtype[Any]") -> "np.ndarray":
    assert _np is not None

    if _np.issubdtype(dtype, _np.integer):
        info = _np.iinfo(dtype)
        clipped = _np.clip(array, info.min, info.max)
        return clipped.astype(dtype)

    if _np.issubdtype(dtype, _np.floating):
        return array.astype(dtype)

    return array.astype(dtype, copy=False)
