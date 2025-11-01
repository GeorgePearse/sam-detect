"""Command line interface for sam-detect."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

from .embedding import AverageColorEmbedder, CLIPEmbedder, Embedder
from .fading import FadeStrategy, GaussianFade, IdentityFade
from .pipeline import SAMDetect, summarize_detection
from .segmentation import NaiveSegmenter, SAM2Segmenter, SegmentationBackend
from .vector_store import InMemoryVectorStore, QdrantVectorStore, VectorStore


def main(argv: Optional[List[str]] = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)

    try:
        segmenter = _build_segmenter(args.segmenter, args.sam_model, args.device)
        embedder = _build_embedder(args.embedder, args.clip_model, args.device)
        fade = _build_fade(args.fade, args.sigma, args.min_fade)
        store = _build_vector_store(args.store, args.qdrant_url, args.qdrant_collection)
    except (ImportError, ValueError) as exc:
        parser.error(str(exc))

    detector = SAMDetect(
        segmenter=segmenter,
        embedder=embedder,
        fade_strategy=fade,
        vector_store=store,
        default_top_k=args.top_k,
    )

    loaded_images = [(path, _load_image(path)) for path in args.images]

    if args.label and loaded_images:
        detector.add_example(loaded_images[0][1], args.label)

    results: List[Dict[str, Any]] = []
    for path, image in loaded_images:
        try:
            detections = detector.detect(image, top_k=args.top_k)
        except (ImportError, NotImplementedError) as exc:
            parser.error(str(exc))
        results.append({"path": str(path), "detections": detections})

    if args.as_json:
        _print_json_output(results)
    else:
        _print_human_output(results)

    return 0


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run the sam-detect pipeline")
    parser.add_argument("images", type=Path, nargs="+", help="Image(s) to process")
    parser.add_argument("--label", help="Optional label to add before detection")

    # Segmentation options
    parser.add_argument(
        "--segmenter",
        choices=("naive", "sam2"),
        default="naive",
        help="Segmentation backend (naive for testing, sam2 for production)",
    )
    parser.add_argument(
        "--sam-model",
        choices=("small", "base", "large"),
        default="base",
        help="SAM2 model size (small/base/large)",
    )

    # Embedding options
    parser.add_argument(
        "--embedder",
        choices=("average", "clip"),
        default="average",
        help="Embedding backend",
    )
    parser.add_argument(
        "--clip-model",
        default="openai/clip-vit-base-patch32",
        help="CLIP model identifier from Hugging Face",
    )

    # Device options
    parser.add_argument(
        "--device",
        choices=("cuda", "cpu"),
        default="cuda",
        help="Device for inference (cuda for GPU, cpu for CPU)",
    )

    # Vector store options
    parser.add_argument(
        "--store",
        choices=("memory", "qdrant"),
        default="memory",
        help="Vector store backend",
    )
    parser.add_argument(
        "--qdrant-url", help="Qdrant endpoint, e.g. http://localhost:6333"
    )
    parser.add_argument(
        "--qdrant-collection",
        default="sam_detect",
        help="Qdrant collection name",
    )

    # Fading options
    parser.add_argument(
        "--fade",
        choices=("identity", "gaussian"),
        default="identity",
        help="Fading strategy",
    )
    parser.add_argument("--sigma", type=float, default=30.0, help="Gaussian sigma")
    parser.add_argument(
        "--min-fade",
        type=float,
        default=0.05,
        help="Minimum fade value for Gaussian strategy",
    )

    # Search options
    parser.add_argument(
        "--top-k",
        type=int,
        default=1,
        help="Number of nearest neighbours to return",
    )

    # Output options
    parser.add_argument(
        "--json",
        dest="as_json",
        action="store_true",
        help="Emit JSON instead of text",
    )
    return parser


def _build_segmenter(choice: str, sam_model: str, device: str) -> SegmentationBackend:
    """Build segmenter based on CLI arguments.

    Args:
        choice: Segmenter choice - "naive" or "sam2"
        sam_model: SAM2 model size - "small", "base", or "large"
        device: Device - "cuda" or "cpu"

    Returns:
        Configured segmentation backend

    Raises:
        ValueError: If invalid segmenter choice
        ImportError: If required dependencies are missing
    """
    if choice == "naive":
        return NaiveSegmenter()
    if choice == "sam2":
        return SAM2Segmenter(model_size=sam_model, device=device)
    raise ValueError(f"Unsupported segmenter '{choice}'")


def _build_embedder(choice: str, model_name: str, device: str) -> Embedder:
    """Build embedder based on CLI arguments.

    Args:
        choice: Embedder choice - "average" or "clip"
        model_name: Model identifier for CLIP
        device: Device - "cuda" or "cpu"

    Returns:
        Configured embedding backend

    Raises:
        ValueError: If invalid embedder choice
        ImportError: If required dependencies are missing
    """
    if choice == "average":
        return AverageColorEmbedder()
    if choice == "clip":
        return CLIPEmbedder(model_name=model_name, device=device)
    raise ValueError(f"Unsupported embedder '{choice}'")


def _build_fade(choice: str, sigma: float, min_fade: float) -> FadeStrategy:
    if choice == "identity":
        return IdentityFade()
    if choice == "gaussian":
        return GaussianFade(sigma=sigma, min_fade=min_fade)
    raise ValueError(f"Unsupported fade strategy '{choice}'")


def _build_vector_store(store: str, url: Optional[str], collection: str) -> VectorStore:
    if store == "memory":
        return InMemoryVectorStore()
    if store == "qdrant":
        return QdrantVectorStore(url=url, collection_name=collection)
    raise ValueError(f"Unsupported vector store '{store}'")


def _print_human_output(results: Iterable[Dict[str, Any]]) -> None:
    for entry in results:
        print(f"Image: {entry['path']}")
        detections = entry["detections"]
        if not detections:
            print("  No detections produced by the segmenter.")
            continue

        for idx, detection in enumerate(detections, start=1):
            summary = summarize_detection(detection["result"])
            print(f"  Detection {idx}:")
            print(f"    label      : {detection['label']}")
            print(f"    confidence : {detection['confidence']:.2f}")
            print(f"    bbox       : {summary['bbox']}")
            print(f"    mask size  : {summary['mask_size']}")
            matches = detection.get("matches", [])
            if matches:
                print("    matches    :")
                for match in matches:
                    print(
                        f"      - {match.get('label', 'unknown')} "
                        f"({match.get('score', 0.0):.2f})"
                    )
            else:
                print("    matches    : none")


def _print_json_output(results: Iterable[Dict[str, Any]]) -> None:
    payload = {
        "images": [
            {
                "path": entry["path"],
                "detections": [_serialize_detection(d) for d in entry["detections"]],
            }
            for entry in results
        ]
    }
    print(json.dumps(payload, indent=2))


def _serialize_detection(detection: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "label": detection.get("label"),
        "confidence": detection.get("confidence"),
        "embedding": _serialize_value(detection.get("embedding")),
        "matches": [_serialize_value(match) for match in detection.get("matches", [])],
        "result": _serialize_value(detection.get("result")),
    }


def _serialize_value(value: Any) -> Any:
    if is_dataclass(value):
        return asdict(value)
    if isinstance(value, dict):
        return {key: _serialize_value(val) for key, val in value.items()}
    if isinstance(value, (list, tuple)):
        return [_serialize_value(item) for item in value]
    if hasattr(value, "tolist"):
        return value.tolist()
    return value


def _load_image(path: Path) -> Any:
    if not path.exists():
        raise FileNotFoundError(path)

    try:
        from PIL import Image  # type: ignore

        return Image.open(path)
    except Exception:
        return path.read_bytes()


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
