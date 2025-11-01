"""Embedding utilities for sam-detect.

This module provides pluggable embedding backends including:
- AverageColorEmbedder: Lightweight baseline using color statistics
- CLIPEmbedder: TensorRT-optimized CLIP vision-language embeddings
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass
from typing import Any, Iterable, Optional, Protocol


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
    """TensorRT-optimized CLIP embedder for vision-language embeddings.

    Uses clip_trt for production-quality CLIP inference with TensorRT acceleration.

    References:
        - clip_trt: https://github.com/dusty-nv/clip_trt
        - CLIP: https://github.com/openai/CLIP
    """

    model_name: str = "openai/clip-vit-base-patch32"
    device: Optional[str] = None
    embedding_dim: int = 512

    def __post_init__(self) -> None:
        """Initialize CLIP model with TensorRT optimization."""
        self._ensure_dependencies()
        self.model = self._load_model()

    def embed(self, image: ImageType) -> Iterable[float]:
        """Generate embedding for an image.

        Args:
            image: Input image (PIL Image, numpy array, or torch tensor)

        Returns:
            Embedding as iterable of floats
        """
        # Get embedding from CLIP model
        embedding = self.model.embed_image(image)

        # Convert to tuple of floats
        if hasattr(embedding, "tolist"):
            # numpy array or torch tensor
            return tuple(float(x) for x in embedding.tolist())
        elif isinstance(embedding, (list, tuple)):
            return tuple(float(x) for x in embedding)

        # Fallback: return zero embedding if something goes wrong
        return tuple([0.0] * self.embedding_dim)

    def _load_model(self) -> Any:
        """Load CLIP model with TensorRT optimization.

        Returns:
            CLIP model with TensorRT enabled

        Raises:
            ImportError: If clip_trt is not installed
        """
        try:
            from clip_trt import CLIPModel
        except ImportError:
            raise ImportError(
                "clip_trt is required for TensorRT-optimized CLIP embeddings. "
                "Install with: pip install 'sam-detect[tensorrt]'"
            )

        import torch

        if self.device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"

        if self.device == "cuda":
            warnings.warn(
                "Using CUDA for CLIP. Ensure CUDA and TensorRT are properly installed for optimal performance.",
                UserWarning,
                stacklevel=2,
            )

        # Load model with TensorRT enabled
        model = CLIPModel.from_pretrained(
            self.model_name,
            use_tensorrt=True,
            device=self.device,
        )

        return model

    def _ensure_dependencies(self) -> None:
        """Verify all required dependencies are installed.

        Raises:
            ImportError: If dependencies are missing
        """
        try:
            import torch  # noqa: F401
        except ImportError as exc:
            raise ImportError(
                "torch is required for CLIP embeddings. "
                "Install with: pip install 'sam-detect[tensorrt]'"
            ) from exc

        try:
            import clip_trt  # noqa: F401
        except ImportError as exc:
            raise ImportError(
                "clip_trt is required for TensorRT-optimized CLIP. "
                "Install with: pip install 'sam-detect[tensorrt]'"
            ) from exc
