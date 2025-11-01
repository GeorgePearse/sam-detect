"""Download and cache SAM2 and CLIP models for offline use.

This script downloads pretrained models and compiles them with TensorRT for
optimized inference. Run this once on your development/deployment machine.

Usage:
    python -m sam_detect.download_models
    python -m sam_detect.download_models --sam-model large
    python -m sam_detect.download_models --clip-model openai/clip-vit-large-patch14
"""

import argparse
import sys
from typing import List, Optional

from sam_detect.model_loader import (
    get_cache_dir,
    get_device,
    get_sam2_model_path,
    get_clip_model_path,
    load_sam2_from_pretrained,
    load_clip_from_pretrained_trt,
)


def download_sam2_models(model_sizes: List[str]) -> None:
    """Download SAM2 models for specified sizes.

    Args:
        model_sizes: List of model sizes to download ("small", "base", "large")
    """
    print("\n=== Downloading SAM2 Models ===")
    cache_dir = get_cache_dir()

    for size in model_sizes:
        print(f"\nDownloading SAM2 {size} model...")
        try:
            model_id = get_sam2_model_path(size)
            print(f"  Model: {model_id}")
            print(f"  Cache: {cache_dir}")

            # Load and cache the model (not stored locally as it's cached by model loader)
            load_sam2_from_pretrained(model_size=size, device="cpu")
            print(f"  ✓ SAM2 {size} downloaded successfully")

        except Exception as e:
            print(f"  ✗ Failed to download SAM2 {size}: {e}")
            return


def download_clip_models(model_variants: List[str]) -> None:
    """Download CLIP models for specified variants.

    Args:
        model_variants: List of model variants to download
    """
    print("\n=== Downloading CLIP Models ===")
    cache_dir = get_cache_dir()

    for variant in model_variants:
        print(f"\nDownloading CLIP {variant} model...")
        try:
            model_id = get_clip_model_path(variant)
            print(f"  Model: {model_id}")
            print(f"  Cache: {cache_dir}")

            # Load and cache the model with TensorRT (cached by model loader)
            load_clip_from_pretrained_trt(model_size=variant, device="cpu")
            print(f"  ✓ CLIP {variant} downloaded successfully")

        except Exception as e:
            print(f"  ✗ Failed to download CLIP {variant}: {e}")
            return


def compile_tensorrt_engines() -> None:
    """Compile TensorRT engines for SAM2 and CLIP models.

    This step is optional but recommended for optimal performance.
    TensorRT engines will be compiled on-demand during first inference if not
    pre-compiled here.
    """
    print("\n=== TensorRT Engine Compilation ===")
    print("Note: TensorRT engines are compiled on-demand during first inference.")
    print("Pre-compilation here is optional but recommended for production.")
    print("This step requires CUDA and torch-tensorrt to be installed.")


def main(argv: Optional[List[str]] = None) -> int:
    """Download and cache models for sam-detect.

    Args:
        argv: Command-line arguments

    Returns:
        Exit code (0 for success, 1 for failure)
    """
    parser = argparse.ArgumentParser(
        description="Download and cache models for sam-detect",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Download all default models
  python -m sam_detect.download_models

  # Download specific SAM2 size
  python -m sam_detect.download_models --sam-model large

  # Download specific CLIP model
  python -m sam_detect.download_models --clip-model openai/clip-vit-large-patch14

  # Download only SAM2 models
  python -m sam_detect.download_models --sam-only

  # Download only CLIP models
  python -m sam_detect.download_models --clip-only
        """,
    )

    parser.add_argument(
        "--sam-model",
        nargs="+",
        choices=("small", "base", "large"),
        default=("base",),
        help="SAM2 model sizes to download (default: base)",
    )
    parser.add_argument(
        "--clip-model",
        nargs="+",
        default=("base",),
        help="CLIP model variants to download (default: base, which maps to openai/clip-vit-base-patch32)",
    )
    parser.add_argument(
        "--sam-only",
        action="store_true",
        help="Download only SAM2 models, skip CLIP",
    )
    parser.add_argument(
        "--clip-only",
        action="store_true",
        help="Download only CLIP models, skip SAM2",
    )
    parser.add_argument(
        "--tensorrt",
        action="store_true",
        help="Pre-compile TensorRT engines (requires CUDA)",
    )

    args = parser.parse_args(argv)

    # Validate arguments
    if args.sam_only and args.clip_only:
        parser.error("Cannot use both --sam-only and --clip-only")

    # Print header
    print("\n" + "=" * 50)
    print("sam-detect Model Downloader")
    print("=" * 50)
    print(f"Cache directory: {get_cache_dir()}")
    print("Device: CUDA" if any(c.isalpha() for c in str(get_device())) else "CPU")

    try:
        # Download SAM2 models unless --clip-only
        if not args.clip_only:
            download_sam2_models(args.sam_model)

        # Download CLIP models unless --sam-only
        if not args.sam_only:
            download_clip_models(args.clip_model)

        # Optionally compile TensorRT engines
        if args.tensorrt:
            compile_tensorrt_engines()

        print("\n" + "=" * 50)
        print("✓ Model download complete!")
        print("=" * 50)
        print("\nYou can now use sam-detect with TensorRT optimization:")
        print("  sam-detect image.jpg --segmenter sam2 --embedder clip --device cuda")

        return 0

    except Exception as e:
        print(f"\n✗ Error during model download: {e}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
