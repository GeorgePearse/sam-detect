"""High level pipeline wiring for sam-detect."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Optional, Sequence

from .embedding import Embedder
from .fading import FadeStrategy
from .model_registry import (
    build_embedder,
    build_fade_strategy,
    build_segmenter,
    build_vector_store,
)
from .segmentation import SegmentationBackend, SegmentationResult
from .vector_store import VectorStore


Detection = Dict[str, Any]
Match = Dict[str, Any]


@dataclass
class SAMDetect:
    """Production-ready SAM2 + CLIP instance detection pipeline.

    This is the main entry point for sam-detect. It provides a simple string-based
    API for configuring and running the detection pipeline.

    Args:
        segmenter: Model identifier for segmentation. Options:
            - "naive": Simple grid-based segmentation (no GPU, no download)
            - "sam2-small": SAM2 Hiera Small (~180MB)
            - "sam2-base": SAM2 Hiera Base (~500MB, default)
            - "sam2-large": SAM2 Hiera Large (~1.2GB)

        embedder: Model identifier for image embedding. Options:
            - "average": Average color embedder (no GPU, no download)
            - "clip-vit-base": CLIP ViT-Base-32 (~100MB, default)
            - "clip-vit-large": CLIP ViT-Large-14 (~400MB)
            - Any HuggingFace CLIP model ID (e.g., "openai/clip-vit-base-patch16")

        fade_strategy: Fading strategy for context preservation. Options:
            - "identity": No fading (use full image)
            - "gaussian": Gaussian fade with defaults (sigma=30, min_fade=0.05)
            - "gaussian-sigma50-min0.2": Custom Gaussian parameters

        vector_store: Vector database backend. Options:
            - "memory": In-memory storage (default, no persistence)
            - "qdrant": Qdrant vector database (requires running server)

        device: Compute device ("cuda" or "cpu"). Default: "cuda"

        qdrant_url: Qdrant server URL. Default: "http://localhost:6333"

        qdrant_collection: Qdrant collection name. Default: "sam_detect"

        default_top_k: Default number of nearest neighbors. Default: 1

    Example:
        >>> # Production-ready defaults
        >>> detector = SAMDetect()

        >>> # Custom configuration
        >>> detector = SAMDetect(
        ...     segmenter="sam2-large",
        ...     embedder="clip-vit-large",
        ...     device="cuda"
        ... )

        >>> # CPU-only, lightweight models
        >>> detector = SAMDetect(
        ...     segmenter="naive",
        ...     embedder="average",
        ...     device="cpu"
        ... )

        >>> # Detect objects
        >>> results = detector.detect("image.jpg")
    """

    # Model configuration (string identifiers)
    segmenter: str = "sam2-base"
    embedder: str = "clip-vit-base"
    fade_strategy: str = "gaussian"
    vector_store: str = "memory"

    # Compute and storage configuration
    device: str = "cuda"
    qdrant_url: Optional[str] = None
    qdrant_collection: str = "sam_detect"
    default_top_k: int = 1

    # Internal objects (populated in __post_init__)
    _segmenter: SegmentationBackend = field(init=False, repr=False)
    _embedder: Embedder = field(init=False, repr=False)
    _fade_strategy: FadeStrategy = field(init=False, repr=False)
    _vector_store: VectorStore = field(init=False, repr=False)

    def __post_init__(self) -> None:
        """Build model objects from string identifiers."""
        # Build each component from its string identifier
        self._segmenter = build_segmenter(self.segmenter, self.device)
        self._embedder = build_embedder(self.embedder, self.device)
        self._fade_strategy = build_fade_strategy(self.fade_strategy)
        self._vector_store = build_vector_store(
            self.vector_store,
            qdrant_url=self.qdrant_url,
            collection_name=self.qdrant_collection,
        )

    def detect(self, image: Any, *, top_k: Optional[int] = None) -> List[Detection]:
        """Run segmentation, embedding, and vector search for an image.

        Args:
            image: Input image (file path, PIL Image, or numpy array)
            top_k: Override default_top_k for this detection

        Returns:
            List of detection dictionaries with segmentation, embedding, and matches
        """
        detections: List[Detection] = []
        k = max(1, int(top_k or self.default_top_k))

        for result in self._segmenter(image):
            faded = self._fade_strategy.apply(image, result)
            vector = tuple(self._embedder.embed(faded))
            raw_matches = self._vector_store.search(vector, k=k)
            matches = [
                {"label": match_label, "score": float(match_score)}
                for match_label, match_score in raw_matches
            ]
            label, confidence = self._resolve_label(matches)
            detections.append(
                {
                    "label": label,
                    "confidence": confidence,
                    "embedding": vector,
                    "matches": matches,
                    "result": result,
                }
            )

        return detections

    def add_example(self, image: Any, label: str) -> tuple[float, ...]:
        """Add a labeled example to the backing vector store.

        Args:
            image: Input image (file path, PIL Image, or numpy array)
            label: Label for this example

        Returns:
            The embedding vector for this image
        """
        embedding = tuple(self._embedder.embed(image))
        self._vector_store.add(embedding, label)
        return embedding

    def add_examples(self, items: Iterable[tuple[Any, str]]) -> None:
        """Add multiple labeled examples to the vector store.

        Args:
            items: Iterable of (image, label) tuples
        """
        for image, label in items:
            self.add_example(image, label)

    @staticmethod
    def _resolve_label(matches: Sequence[Match]) -> tuple[str, float]:
        if not matches:
            return "unknown", 0.0
        top = matches[0]
        label = str(top.get("label", "unknown"))
        score_value = top.get("score", 0.0)
        try:
            score = float(score_value)
        except (TypeError, ValueError):
            score = 0.0
        return label, score


def summarize_detection(result: SegmentationResult) -> Dict[str, Any]:
    """Convenience helper for quick logging and tests."""

    return {"bbox": result.bbox, "mask_size": (len(result.mask), len(result.mask[0]))}
