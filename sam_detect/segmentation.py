"""Segmentation backends for sam-detect.

This module provides pluggable segmentation backends including:
- NaiveSegmenter: Lightweight fallback for testing
- SAM2Segmenter: Production SAM2 with automatic mask generation

The interface is lightweight so backends can be easily swapped.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass
from typing import Any, Iterable, Optional, Protocol, Sequence

import numpy as np


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


def _normalize_image_to_numpy(image: ImageType) -> np.ndarray:
    """Convert image to numpy array in RGB format.

    Args:
        image: Pillow Image, numpy array, or torch tensor

    Returns:
        Image as numpy array in RGB format with shape (H, W, 3)
    """
    import torch

    # Handle torch tensors
    if isinstance(image, torch.Tensor):
        image = image.cpu().numpy()

    # Handle Pillow Images
    if hasattr(image, "convert"):
        image = image.convert("RGB")
        image = np.array(image)

    # Ensure uint8
    if image.dtype != np.uint8:
        if image.max() <= 1.0:
            image = (image * 255).astype(np.uint8)
        else:
            image = image.astype(np.uint8)

    return image


class SAM2Segmenter:
    """SAM2 segmentation backend with automatic mask generation.

    Uses SAM2's automatic mask generation to segment all objects in an image
    without requiring user-provided prompts.

    References:
        - SAM2: https://github.com/facebookresearch/segment-anything-2
        - TensorRT optimization: https://docs.pytorch.org/TensorRT/tutorials/_rendered_examples/dynamo/torch_export_sam2.html
    """

    def __init__(
        self,
        model_size: str = "base",
        device: Optional[str] = None,
        points_per_side: int = 32,
        pred_iou_thresh: float = 0.88,
    ) -> None:
        """Initialize SAM2 segmenter with automatic mask generation.

        Args:
            model_size: SAM2 model size - "small", "base", or "large"
            device: Device to load model on. If None, uses CUDA if available
            points_per_side: Number of points per side in automatic mask generation grid
            pred_iou_thresh: Confidence threshold for mask predictions
        """
        try:
            from sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator
            from sam2.build_sam import build_sam2
        except ImportError:
            raise ImportError(
                "segment-anything-2 is required for SAM2 segmentation. "
                "Install with: pip install 'sam-detect[tensorrt]'"
            )

        import torch

        # Determine device
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
            if device == "cuda":
                warnings.warn(
                    "Using CUDA for SAM2. For TensorRT optimization, ensure CUDA and TensorRT are properly installed.",
                    UserWarning,
                    stacklevel=2,
                )

        self.device = device
        self.model_size = model_size

        # Load SAM2 model
        # Note: SAM2 uses yaml config files for different model sizes
        self.model = build_sam2(
            config_file=f"sam2_hiera_{model_size}.yaml",
            ckpt_path=None,  # Auto-download from huggingface
            device=device,
        )

        # Create automatic mask generator
        self.mask_generator = SAM2AutomaticMaskGenerator(
            model=self.model,
            points_per_side=points_per_side,
            pred_iou_thresh=pred_iou_thresh,
        )

    def __call__(self, image: ImageType) -> Iterable[SegmentationResult]:
        """Generate masks for all objects in image.

        Args:
            image: Input image (Pillow Image, numpy array, or torch tensor)

        Yields:
            SegmentationResult for each detected object
        """
        # Convert to numpy array
        image_np = _normalize_image_to_numpy(image)

        # Get original dimensions for later reference
        height, width = image_np.shape[:2]

        # Generate masks
        masks = self.mask_generator.generate(image_np)

        # Convert each mask to SegmentationResult
        for mask_dict in masks:
            # SAM2AutomaticMaskGenerator returns dict with 'segmentation' key
            binary_mask = mask_dict["segmentation"]  # Shape: (H, W)

            # Convert to list of lists format for compatibility
            mask_list = [
                [bool(binary_mask[y, x]) for x in range(width)] for y in range(height)
            ]

            # Get bounding box from mask
            bbox = self._mask_to_bbox(binary_mask)

            yield SegmentationResult(mask=mask_list, bbox=bbox)

    @staticmethod
    def _mask_to_bbox(mask: np.ndarray) -> tuple[int, int, int, int]:
        """Convert binary mask to bounding box.

        Args:
            mask: Binary mask array with shape (H, W)

        Returns:
            Bounding box as (x1, y1, x2, y2) in pixel coordinates
        """
        rows = np.any(mask, axis=1)
        cols = np.any(mask, axis=0)

        if not rows.any() or not cols.any():
            return (0, 0, 1, 1)  # Empty mask fallback

        y_min, y_max = np.where(rows)[0][[0, -1]]
        x_min, x_max = np.where(cols)[0][[0, -1]]

        return (int(x_min), int(y_min), int(x_max) + 1, int(y_max) + 1)
