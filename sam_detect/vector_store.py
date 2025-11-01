"""Vector store abstractions."""

from __future__ import annotations

import importlib
import logging
import math
import uuid
from dataclasses import dataclass, field
from typing import Any, Iterable, List, Optional, Protocol, Sequence


Vector = Sequence[float]

_qdrant_client_module = None
_qdrant_models_module = None

try:  # pragma: no cover - optional dependency
    _qdrant_client_module = importlib.import_module("qdrant_client")
    _qdrant_models_module = importlib.import_module("qdrant_client.http.models")
except Exception:  # pragma: no cover - qdrant is optional
    _qdrant_client_module = None
    _qdrant_models_module = None

QdrantClient = (
    None
    if _qdrant_client_module is None
    else getattr(_qdrant_client_module, "QdrantClient", None)
)
qmodels = _qdrant_models_module


logger = logging.getLogger(__name__)


class VectorStore(Protocol):
    """Interface for feature vector persistence."""

    def add(self, vector: Vector, label: str) -> None: ...

    def search(self, query: Vector, k: int = 5) -> List[tuple[str, float]]: ...


@dataclass
class InMemoryVectorStore:
    """Simple list-backed vector store for smoke tests."""

    items: List[tuple[Vector, str]] = field(default_factory=list)

    def add(self, vector: Vector, label: str) -> None:
        self.items.append((tuple(vector), label))

    def search(self, query: Vector, k: int = 5) -> List[tuple[str, float]]:
        query_tuple = tuple(query)
        scored = [
            (label, _cosine_similarity(query_tuple, stored))
            for stored, label in self.items
        ]
        scored.sort(key=lambda item: item[1], reverse=True)
        return scored[:k]


@dataclass
class QdrantVectorStore:
    """Qdrant-backed vector store. Requires ``qdrant-client``."""

    collection_name: str = "sam_detect"
    url: Optional[str] = None
    api_key: Optional[str] = None
    client_kwargs: Optional[dict[str, Any]] = None

    _client: Any = field(init=False, repr=False)
    _vector_size: Optional[int] = field(default=None, init=False, repr=False)
    _collection_checked: bool = field(default=False, init=False, repr=False)

    def __post_init__(self) -> None:
        if QdrantClient is None or qmodels is None:
            raise ImportError(
                "QdrantVectorStore requires the `qdrant-client` package. "
                "Install extras via `pip install sam-detect[qdrant]`."
            )

        kwargs: dict[str, Any] = dict(self.client_kwargs or {})
        if self.url:
            kwargs.setdefault("url", self.url)
        if self.api_key:
            kwargs.setdefault("api_key", self.api_key)
        self._client = QdrantClient(**kwargs)

    def add(self, vector: Vector, label: str) -> None:
        floats = [float(v) for v in vector]
        if not floats:
            raise ValueError("Vectors must contain at least one value")

        self._ensure_collection(len(floats))

        assert qmodels is not None
        point = qmodels.PointStruct(
            id=uuid.uuid4().hex,
            vector=floats,
            payload={"label": label},
        )

        try:
            self._client.upsert(collection_name=self.collection_name, points=[point])
        except Exception as exc:  # pragma: no cover - network dependent
            logger.warning("Failed to upsert vector into Qdrant: %s", exc)
            raise

    def search(self, query: Vector, k: int = 5) -> List[tuple[str, float]]:
        floats = [float(v) for v in query]
        if not floats:
            return []

        self._ensure_collection(len(floats))

        try:
            results = self._client.search(
                collection_name=self.collection_name,
                query_vector=floats,
                limit=k,
                with_payload=True,
            )
        except Exception as exc:  # pragma: no cover - network dependent
            logger.warning("Failed to search Qdrant: %s", exc)
            return []

        matches: List[tuple[str, float]] = []
        for item in results:
            payload = getattr(item, "payload", {}) or {}
            label = payload.get("label", "unknown")
            score = float(getattr(item, "score", 0.0))
            matches.append((label, score))

        return matches

    def _ensure_collection(self, size: int) -> None:
        if self._vector_size is None:
            self._vector_size = size
        elif self._vector_size != size:
            raise ValueError(
                f"Qdrant collection expects vectors of size {self._vector_size}, "
                f"received {size}"
            )

        if self._collection_checked:
            return

        assert qmodels is not None
        try:
            self._client.get_collection(self.collection_name)
        except Exception:  # pragma: no cover - network dependent
            vector_params = qmodels.VectorParams(
                size=size,
                distance=qmodels.Distance.COSINE,
            )
            try:
                self._client.create_collection(
                    collection_name=self.collection_name,
                    vectors_config=vector_params,
                )
            except Exception as exc:
                logger.warning("Failed to create Qdrant collection: %s", exc)
                raise

        self._collection_checked = True


def _cosine_similarity(a: Iterable[float], b: Iterable[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(y * y for y in b))
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot / (norm_a * norm_b)
