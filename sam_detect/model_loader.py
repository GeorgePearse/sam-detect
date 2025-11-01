"""Model loading and caching utilities for TensorRT-optimized SAM2 and CLIP."""

import os
from pathlib import Path
from typing import Optional

import torch


def get_cache_dir() -> Path:
    """Get the cache directory for sam-detect models.

    Returns:
        Path to cache directory, defaults to ~/.cache/sam-detect/
    """
    cache_dir = Path(
        os.environ.get("SAM_DETECT_CACHE", Path.home() / ".cache" / "sam-detect")
    )
    cache_dir.mkdir(parents=True, exist_ok=True)
    return cache_dir


def get_device() -> str:
    """Get the appropriate device for TensorRT inference.

    Returns:
        Device string: "cuda" if available, otherwise "cpu"

    Raises:
        RuntimeError: If TensorRT is requested but CUDA is not available
    """
    if not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA is required for TensorRT-optimized SAM2 and CLIP. "
            "Please ensure you have a CUDA-compatible GPU and NVIDIA drivers installed."
        )
    return "cuda"


def get_sam2_model_path(model_size: str = "base") -> str:
    """Get Hugging Face model ID for SAM2.

    Args:
        model_size: Model size - "small", "base", or "large"

    Returns:
        Hugging Face model ID

    Raises:
        ValueError: If invalid model size
    """
    valid_sizes = {"small", "base", "large"}
    if model_size not in valid_sizes:
        raise ValueError(
            f"Invalid SAM2 model size '{model_size}'. " f"Must be one of: {valid_sizes}"
        )

    return f"facebook/sam2-hiera-{model_size}"


def get_clip_model_path(model_size: str = "base") -> str:
    """Get Hugging Face model ID for CLIP.

    Args:
        model_size: Model variant - "base", "large", etc.

    Returns:
        Hugging Face model ID
    """
    # Support common CLIP model variants
    model_map = {
        "base": "openai/clip-vit-base-patch32",
        "large": "openai/clip-vit-large-patch14",
        "base-patch16": "openai/clip-vit-base-patch16",
        "large-patch14-336": "openai/clip-vit-large-patch14-336",
    }

    if model_size in model_map:
        return model_map[model_size]

    # If not in map, assume it's a full model ID
    return model_size


def load_sam2_from_pretrained(
    model_size: str = "base",
    device: Optional[str] = None,
) -> torch.nn.Module:
    """Load SAM2 model from Hugging Face.

    Args:
        model_size: Model size - "small", "base", or "large"
        device: Device to load model on. If None, uses auto-detected CUDA device

    Returns:
        SAM2 model loaded and ready for inference

    Raises:
        ImportError: If segment_anything_2 is not installed
        RuntimeError: If device is CUDA but not available
    """
    try:
        from sam2.build_sam import build_sam2
    except ImportError:
        raise ImportError(
            "segment-anything-2 is required for SAM2 integration. "
            "Install with: pip install 'sam-detect[tensorrt]'"
        )

    if device is None:
        device = get_device()

    cache_dir = get_cache_dir()

    # SAM2 model building
    # Note: SAM2 downloads weights automatically with huggingface_hub
    # We just need to build and move to device
    model = build_sam2(
        config_file=f"sam2_hiera_{model_size}.yaml",
        ckpt_path=str(cache_dir / f"sam2_hiera_{model_size}.pt"),
        device=device,
    )

    return model


def load_clip_from_pretrained_trt(
    model_size: str = "base",
    device: Optional[str] = None,
) -> torch.nn.Module:
    """Load CLIP model with TensorRT optimization from clip_trt.

    Args:
        model_size: Model variant - "base", "large", etc.
        device: Device to load model on (TensorRT requires CUDA)

    Returns:
        CLIP model with TensorRT optimization enabled

    Raises:
        ImportError: If clip_trt is not installed
        RuntimeError: If device is CUDA but not available
    """
    try:
        from clip_trt import CLIPModel
    except ImportError:
        raise ImportError(
            "clip_trt is required for optimized CLIP embedding. "
            "Install with: pip install 'sam-detect[tensorrt]'"
        )

    if device is None:
        device = get_device()

    model_id = get_clip_model_path(model_size)
    cache_dir = get_cache_dir()

    # Load CLIP model with TensorRT enabled
    model = CLIPModel.from_pretrained(
        model_id,
        use_tensorrt=True,
        cache_dir=str(cache_dir),
        device=device,
    )

    return model


def get_tensorrt_engine_path(model_name: str, model_size: str) -> Path:
    """Get the path where TensorRT compiled engine should be cached.

    Args:
        model_name: Name of model ("sam2" or "clip")
        model_size: Model size/variant

    Returns:
        Path to TensorRT engine file
    """
    cache_dir = get_cache_dir()
    engine_dir = cache_dir / "tensorrt_engines"
    engine_dir.mkdir(parents=True, exist_ok=True)
    return engine_dir / f"{model_name}_{model_size}.pt"


def cache_available() -> bool:
    """Check if cached models are already available.

    Returns:
        True if cache directory exists with expected structure
    """
    cache_dir = get_cache_dir()
    return cache_dir.exists()
