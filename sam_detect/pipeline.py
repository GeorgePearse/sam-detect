"""High level pipeline wiring for sam-detect."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Optional, Sequence

from .embedding import AverageColorEmbedder, Embedder
from .fading import FadeStrategy, IdentityFade
from .segmentation import NaiveSegmenter, SegmentationBackend, SegmentationResult
from .vector_store import InMemoryVectorStore, VectorStore


Detection = Dict[str, Any]
Match = Dict[str, Any]


@dataclass
class SAMDetect:
    """Composable detection pipeline with sensible defaults."""

    segmenter: SegmentationBackend = field(default_factory=NaiveSegmenter)
    fade_strategy: FadeStrategy = field(default_factory=IdentityFade)
    embedder: Embedder = field(default_factory=AverageColorEmbedder)
    vector_store: VectorStore = field(default_factory=InMemoryVectorStore)
    default_top_k: int = 1

    def detect(self, image: Any, *, top_k: Optional[int] = None) -> List[Detection]:
        """Run segmentation, embedding, and vector search for an image."""

        detections: List[Detection] = []
        k = max(1, int(top_k or self.default_top_k))

        for result in self.segmenter(image):
            faded = self.fade_strategy.apply(image, result)
            vector = tuple(self.embedder.embed(faded))
            raw_matches = self.vector_store.search(vector, k=k)
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
        """Add a labeled example to the backing vector store."""

        embedding = tuple(self.embedder.embed(image))
        self.vector_store.add(embedding, label)
        return embedding

    def add_examples(self, items: Iterable[tuple[Any, str]]) -> None:
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
