#!/usr/bin/env python3
"""Test pipeline for sam-detect on COCO dataset."""

import json
import sys
from pathlib import Path
from typing import Any, Dict, Union


from sam_detect import SAMDetect
from sam_detect.coco_utils import COCODataset
from sam_detect.embedding import AverageColorEmbedder
from sam_detect.fading import IdentityFade
from sam_detect.segmentation import NaiveSegmenter
from sam_detect.vector_store import InMemoryVectorStore


def run_test_pipeline(
    dataset_dir: Union[str, Path] = "./datasets/coco-minitrain",
    num_images: int = 10,
    num_examples: int = 3,
) -> Dict[str, Any]:
    """
    Run test pipeline: detect on images, add examples, evaluate.

    Args:
        dataset_dir: Path to COCO dataset
        num_images: Number of images to test on
        num_examples: Number of labeled examples to add before detection

    Returns:
        Dictionary with results and metrics
    """
    dataset_dir = Path(dataset_dir)

    # Load dataset
    print("Loading COCO dataset...")
    images_dir = dataset_dir / "val2017"
    annotations_file = dataset_dir / "annotations" / "instances_val2017.json"

    if not images_dir.exists() or not annotations_file.exists():
        print(f"Error: Dataset not found at {dataset_dir}")
        print(f"  Images: {images_dir.exists()}")
        print(f"  Annotations: {annotations_file.exists()}")
        return {"error": "Dataset not found"}

    dataset = COCODataset(images_dir, annotations_file)
    info = dataset.get_dataset_info()

    print("\nDataset Info:")
    print(f"  Total images: {info['num_images']}")
    print(f"  Total annotations: {info['num_annotations']}")
    print(f"  Categories: {info['num_categories']}")
    print(f"  Sample classes: {', '.join(info['categories'][:5])}...")

    # Initialize detector
    print("\nInitializing SAMDetect...")
    detector = SAMDetect(
        segmenter=NaiveSegmenter(),
        fade_strategy=IdentityFade(),
        embedder=AverageColorEmbedder(),
        vector_store=InMemoryVectorStore(),
        default_top_k=3,
    )

    # Phase 1: Add labeled examples
    print(f"\nPhase 1: Adding {num_examples} labeled examples...")
    example_results = []

    for idx, (image_id, image, annotations) in enumerate(dataset):
        if idx >= num_examples:
            break

        labels = dataset.get_annotation_labels(annotations)
        if not labels:
            continue

        # Add examples for each detected category
        for label in set(labels):
            try:
                embedding = detector.add_example(image, label)
                example_results.append(
                    {
                        "image_id": image_id,
                        "label": label,
                        "embedding_dim": len(embedding),
                    }
                )
                print(f"  ✓ Added example: {label} (image {image_id})")
            except Exception as e:
                print(f"  ✗ Failed to add example {label}: {e}")

    print(f"Added {len(example_results)} examples")

    # Phase 2: Run detection on test images
    print(f"\nPhase 2: Running detection on {num_images} images...")
    detection_results = []
    detection_stats = {
        "total_images": 0,
        "total_detections": 0,
        "total_objects": 0,
        "avg_detections_per_image": 0.0,
        "images_with_detections": 0,
    }

    for idx, (image_id, image, annotations) in enumerate(dataset):
        if idx >= num_images:
            break

        detection_stats["total_images"] += 1
        ground_truth_labels = dataset.get_annotation_labels(annotations)

        try:
            detections = detector.detect(image, top_k=3)
            detection_stats["total_detections"] += len(detections)
            detection_stats["total_objects"] += len(ground_truth_labels)

            if detections:
                detection_stats["images_with_detections"] += 1

            image_result = {
                "image_id": image_id,
                "num_detections": len(detections),
                "ground_truth_labels": ground_truth_labels,
                "predictions": [
                    {
                        "label": d["label"],
                        "confidence": d["confidence"],
                        "matches": d.get("matches", [])[:1],  # Top match only
                    }
                    for d in detections
                ],
            }
            detection_results.append(image_result)

            progress = f"[{idx + 1}/{num_images}]"
            labels_str = ", ".join(ground_truth_labels[:3])
            print(
                f"  {progress} Image {image_id}: {len(detections)} detections | GT: {labels_str}"
            )

        except Exception as e:
            print(f"  [Error] Image {image_id}: {e}")

    # Compute statistics
    if detection_stats["total_images"] > 0:
        detection_stats["avg_detections_per_image"] = (
            detection_stats["total_detections"] / detection_stats["total_images"]
        )

    # Prepare results
    results = {
        "dataset_info": info,
        "examples_added": len(example_results),
        "detection_stats": detection_stats,
        "sample_detections": detection_results[:3],  # Show first 3 as examples
        "all_detections": detection_results,
    }

    return results


def print_results(results: Dict[str, Any]) -> None:
    """Print test results in human-readable format."""
    if "error" in results:
        print(f"\nError: {results['error']}")
        return

    print("\n" + "=" * 70)
    print("TEST PIPELINE RESULTS")
    print("=" * 70)

    # Dataset info
    info = results["dataset_info"]
    print("\nDataset: COCO val2017")
    print(f"  Total images: {info['num_images']:,}")
    print(f"  Total annotations: {info['num_annotations']:,}")
    print(f"  Categories: {info['num_categories']}")

    # Examples added
    print("\nTraining Phase:")
    print(f"  Examples added: {results['examples_added']}")

    # Detection stats
    stats = results["detection_stats"]
    print("\nDetection Phase:")
    print(f"  Images processed: {stats['total_images']}")
    print(f"  Total detections: {stats['total_detections']}")
    print(f"  Ground truth objects: {stats['total_objects']}")
    print(f"  Images with detections: {stats['images_with_detections']}")
    print(f"  Avg detections/image: {stats['avg_detections_per_image']:.2f}")

    if stats["total_detections"] > 0:
        detection_rate = (
            stats["images_with_detections"] / stats["total_images"] * 100
            if stats["total_images"] > 0
            else 0
        )
        print(f"  Detection rate: {detection_rate:.1f}%")

    # Sample detections
    print("\nSample Detections (first 3 images):")
    for det in results["sample_detections"]:
        print(f"\n  Image {det['image_id']}:")
        print(f"    Detections: {det['num_detections']}")
        print(f"    Ground truth: {', '.join(det['ground_truth_labels'])}")
        if det["predictions"]:
            for pred in det["predictions"][:2]:
                match_str = ""
                if pred["matches"]:
                    match = pred["matches"][0]
                    match_str = f" (matched: {match.get('label', '?')} @ {match.get('score', 0):.2f})"
                print(f"      - {pred['label']} ({pred['confidence']:.2f}){match_str}")

    print("\n" + "=" * 70)


def save_results(
    results: Dict[str, Any], output_file: str = "test_results.json"
) -> None:
    """Save results to JSON file."""
    output_path = Path(output_file)
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to: {output_path}")


def main() -> int:
    """Run the test pipeline."""
    dataset_dir = "./datasets/coco-minitrain"

    print("SAMDetect Test Pipeline")
    print("=" * 70)

    # Run pipeline
    results = run_test_pipeline(
        dataset_dir=dataset_dir,
        num_images=10,
        num_examples=3,
    )

    # Print results
    print_results(results)

    # Save results
    save_results(results)

    return 0 if "error" not in results else 1


if __name__ == "__main__":
    sys.exit(main())
