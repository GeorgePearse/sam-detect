"""Model registry and builders for string-based API.

This module provides functions to build segmenters, embedders, fade strategies,
and vector stores from string identifiers. It centralizes model creation logic
that's shared between the Python API and CLI.
"""

from typing import Optional

from .embedding import Embedder
from .fading import FadeStrategy
from .segmentation import SegmentationBackend
from .vector_store import VectorStore

# Supported model identifiers
SEGMENTERS = ["naive", "sam2-small", "sam2-base", "sam2-large"]
EMBEDDERS = ["average", "clip-vit-base", "clip-vit-large"]
FADE_STRATEGIES = ["identity", "gaussian"]
VECTOR_STORES = ["memory", "qdrant"]

# Model aliases and mappings
CLIP_ALIASES = {
    "clip-vit-base": "openai/clip-vit-base-patch32",
    "clip-vit-large": "openai/clip-vit-large-patch14",
}

# SAM2 model size validation
SAM2_SIZES = ["small", "base", "large"]


def build_segmenter(model_str: str, device: str) -> SegmentationBackend:
    """Build segmenter from string identifier.

    Args:
        model_str: Model identifier. Options:
            - "naive": Simple grid-based segmentation
            - "sam2-small": SAM2 Hiera Small
            - "sam2-base": SAM2 Hiera Base
            - "sam2-large": SAM2 Hiera Large

        device: Compute device ("cuda" or "cpu")

    Returns:
        SegmentationBackend instance ready for inference

    Raises:
        ValueError: If model_str is unknown or invalid
    """
    if model_str == "naive":
        from .segmentation import NaiveSegmenter

        return NaiveSegmenter()

    if model_str.startswith("sam2-"):
        from .segmentation import SAM2Segmenter

        # Extract size from "sam2-base" -> "base"
        size = model_str.split("-", 1)[1]
        if size not in SAM2_SIZES:
            raise ValueError(f"Unknown SAM2 size: {size}. Options: {SAM2_SIZES}")
        return SAM2Segmenter(model_size=size, device=device)

    raise ValueError(f"Unknown segmenter: {model_str}. Options: {SEGMENTERS}")


def build_embedder(model_str: str, device: str) -> Embedder:
    """Build embedder from string identifier.

    Args:
        model_str: Model identifier. Options:
            - "average": Average color embedder (simple, no GPU required)
            - "clip-vit-base": CLIP ViT-Base-32
            - "clip-vit-large": CLIP ViT-Large-14
            - Any HuggingFace CLIP model ID (e.g., "openai/clip-vit-base-patch16")

        device: Compute device ("cuda" or "cpu")

    Returns:
        Embedder instance ready for inference

    Raises:
        ValueError: If model_str is unknown or invalid
    """
    if model_str == "average":
        from .embedding import AverageColorEmbedder

        return AverageColorEmbedder()

    # Check if it's an alias
    model_name = CLIP_ALIASES.get(model_str, model_str)

    # Assume it's a CLIP model (could be alias or full HuggingFace ID)
    if model_str.startswith("clip-") or "/" in model_name:
        from .embedding import CLIPEmbedder

        return CLIPEmbedder(model_name=model_name, device=device)

    raise ValueError(
        f"Unknown embedder: {model_str}. Options: {EMBEDDERS} or HuggingFace model IDs"
    )


def build_fade_strategy(fade_str: str) -> FadeStrategy:
    """Build fade strategy from string identifier.

    Args:
        fade_str: Fade strategy identifier. Examples:
            - "identity": No fading (use full image context)
            - "gaussian": Gaussian fade with defaults (sigma=30, min_fade=0.05)
            - "gaussian-sigma50": Gaussian with custom sigma
            - "gaussian-sigma50-min0.2": Gaussian with custom sigma and min

        Returns:
        FadeStrategy instance

    Raises:
        ValueError: If fade_str is unknown or invalid
    """
    if fade_str == "identity":
        from .fading import IdentityFade

        return IdentityFade()

    if fade_str.startswith("gaussian"):
        from .fading import GaussianFade

        # Parse "gaussian", "gaussian-sigma30", "gaussian-sigma30-min0.1"
        parts = fade_str.split("-")
        kwargs = {}

        for part in parts[1:]:
            if part.startswith("sigma"):
                try:
                    kwargs["sigma"] = float(part[5:])
                except ValueError:
                    raise ValueError(f"Invalid sigma value in fade_str: {fade_str}")
            elif part.startswith("min"):
                try:
                    kwargs["min_fade"] = float(part[3:])
                except ValueError:
                    raise ValueError(f"Invalid min_fade value in fade_str: {fade_str}")

        return GaussianFade(**kwargs)

    raise ValueError(f"Unknown fade strategy: {fade_str}. Options: {FADE_STRATEGIES}")


def build_vector_store(
    store_str: str,
    qdrant_url: Optional[str] = None,
    collection_name: str = "sam_detect",
) -> VectorStore:
    """Build vector store from string identifier.

    Args:
        store_str: Vector store type. Options:
            - "memory": In-memory storage (no persistence)
            - "qdrant": Qdrant vector database

        qdrant_url: Qdrant server URL (only needed if store_str=="qdrant")
            Default: "http://localhost:6333"

        collection_name: Qdrant collection name. Default: "sam_detect"

    Returns:
        VectorStore instance

    Raises:
        ValueError: If store_str is unknown or invalid
    """
    if store_str == "memory":
        from .vector_store import InMemoryVectorStore

        return InMemoryVectorStore()

    if store_str == "qdrant":
        from .vector_store import QdrantVectorStore

        url = qdrant_url or "http://localhost:6333"
        return QdrantVectorStore(collection_name=collection_name, url=url)

    raise ValueError(f"Unknown vector store: {store_str}. Options: {VECTOR_STORES}")
