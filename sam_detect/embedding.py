"""Embedding utilities for sam-detect."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable, Protocol

ImageType = Any


class Embedder(Protocol):
    """Protocol for producing dense feature vectors from images."""

    def embed(self, image: ImageType) -> Iterable[float]: ...


@dataclass
class AverageColorEmbedder:
    """Simple baseline embedder based on average RGB values."""

    def embed(self, image: ImageType) -> Iterable[float]:
        if hasattr(image, "convert"):
            image = image.convert("RGB")
            pixels = list(image.getdata())
            if not pixels:
                return (0.0, 0.0, 0.0)
            r = sum(p[0] for p in pixels) / len(pixels)
            g = sum(p[1] for p in pixels) / len(pixels)
            b = sum(p[2] for p in pixels) / len(pixels)
            return (r, g, b)

        if hasattr(image, "mean"):
            # NumPy arrays expose mean
            mean_value = image.mean(axis=(0, 1))
            if isinstance(mean_value, (int, float)):
                return (float(mean_value),) * 3
            return tuple(float(x) for x in mean_value[:3])

        return (0.0, 0.0, 0.0)


@dataclass
class CLIPEmbedder:
    """Stub embedder guiding users toward a CLIP implementation."""

    model_name: str = "ViT-B-32"
    device: str | None = None

    def __post_init__(self) -> None:
        self._ensure_dependencies()

    def embed(self, image: ImageType) -> Iterable[float]:
        raise NotImplementedError(
            "CLIPEmbedder is a placeholder. Provide an implementation that "
            "loads CLIP weights and returns an embedding vector."
        )

    def _ensure_dependencies(self) -> None:
        try:  # pragma: no cover - optional dependency check
            import torch  # type: ignore  # noqa: F401
        except Exception as exc:  # pragma: no cover
            raise ImportError(
                "CLIPEmbedder requires `torch`. Install extras via "
                "`pip install sam-detect[clip]`."
            ) from exc

        try:  # pragma: no cover - optional dependency check
            import open_clip  # type: ignore  # noqa: F401
        except Exception as exc:  # pragma: no cover
            raise ImportError(
                "CLIPEmbedder requires `open_clip_torch`. Install extras via "
                "`pip install sam-detect[clip]`."
            ) from exc
