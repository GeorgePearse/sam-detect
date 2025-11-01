"""Export SAM2 and CLIP models to ONNX format for use with ort (ONNX Runtime).

This script exports:
- SAM2 Hiera Base model with automatic mask generation support
- CLIP ViT-Base-32 vision model for embeddings

Output: rust-cli/models/{sam2_base.onnx, clip_vit_base.onnx}
"""

import argparse
import sys
from pathlib import Path

import torch
import torch.onnx


def export_sam2_onnx(output_path: str = "models/sam2_base.onnx") -> None:
    """Export SAM2 Hiera Base model to ONNX format.

    Args:
        output_path: Where to save the ONNX model

    Raises:
        ImportError: If segment-anything-2 is not installed
    """
    print("=" * 60)
    print("Exporting SAM2 Hiera Base to ONNX")
    print("=" * 60)

    try:
        from sam2.build_sam import build_sam2
    except ImportError:
        print("ERROR: segment-anything-2 is required for SAM2 export")
        print("Install with: pip install 'sam-detect[tensorrt]'")
        sys.exit(1)

    # Build SAM2 model
    print("Building SAM2 model...")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = build_sam2("sam2_hiera_base.yaml", device=device)
    model.eval()

    # Create dummy input
    print("Creating dummy input (1024x1024 RGB image)...")
    dummy_input = torch.randn(1, 3, 1024, 1024, device=device, dtype=torch.float32)

    # Export to ONNX
    print(f"Exporting to ONNX (device: {device})...")
    try:
        # For SAM2 automatic mask generator, we export the image encoder
        # The mask decoder is more complex and typically handled separately
        torch.onnx.export(
            model.image_encoder,
            dummy_input,
            output_path,
            input_names=["image"],
            output_names=["image_embeddings"],
            dynamic_axes={
                "image": {0: "batch", 2: "height", 3: "width"},
                "image_embeddings": {0: "batch"},
            },
            opset_version=17,
            verbose=False,
            do_constant_folding=True,
        )
        print(f"✓ SAM2 exported successfully to {output_path}")
        print(f"  File size: {Path(output_path).stat().st_size / 1024 / 1024:.1f} MB")
    except Exception as e:
        print(f"✗ Failed to export SAM2: {e}")
        raise


def export_clip_onnx(output_path: str = "models/clip_vit_base.onnx") -> None:
    """Export CLIP ViT-Base-32 vision model to ONNX format.

    Args:
        output_path: Where to save the ONNX model

    Raises:
        ImportError: If transformers is not installed
    """
    print("\n" + "=" * 60)
    print("Exporting CLIP ViT-Base-32 to ONNX")
    print("=" * 60)

    try:
        from transformers import CLIPModel
    except ImportError:
        print("ERROR: transformers is required for CLIP export")
        print("Install with: pip install transformers")
        sys.exit(1)

    # Load CLIP model
    print("Loading CLIP model...")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
    model.to(device)
    model.eval()

    # Create dummy input (224x224 RGB image)
    print("Creating dummy input (224x224 RGB image)...")
    dummy_input = torch.randn(1, 3, 224, 224, device=device, dtype=torch.float32)

    # Export vision model only (for embeddings)
    print(f"Exporting to ONNX (device: {device})...")
    try:
        torch.onnx.export(
            model.vision_model,
            dummy_input,
            output_path,
            input_names=["pixel_values"],
            output_names=["embeddings"],
            dynamic_axes={
                "pixel_values": {0: "batch", 2: "height", 3: "width"},
                "embeddings": {0: "batch"},
            },
            opset_version=17,
            verbose=False,
            do_constant_folding=True,
        )
        print(f"✓ CLIP exported successfully to {output_path}")
        print(f"  File size: {Path(output_path).stat().st_size / 1024 / 1024:.1f} MB")
    except Exception as e:
        print(f"✗ Failed to export CLIP: {e}")
        raise


def main() -> int:
    """Main entry point for ONNX export script.

    Returns:
        Exit code (0 for success, 1 for failure)
    """
    parser = argparse.ArgumentParser(
        description="Export SAM2 and CLIP models to ONNX format",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Export both models to default locations
  python scripts/export_models.py

  # Export only SAM2
  python scripts/export_models.py --sam2-only

  # Export to custom location
  python scripts/export_models.py --sam2-output /path/to/sam2.onnx
        """,
    )

    parser.add_argument(
        "--sam2-output",
        default="models/sam2_base.onnx",
        help="Path to save SAM2 ONNX model",
    )
    parser.add_argument(
        "--clip-output",
        default="models/clip_vit_base.onnx",
        help="Path to save CLIP ONNX model",
    )
    parser.add_argument(
        "--sam2-only",
        action="store_true",
        help="Export only SAM2 model",
    )
    parser.add_argument(
        "--clip-only",
        action="store_true",
        help="Export only CLIP model",
    )

    args = parser.parse_args()

    # Validate arguments
    if args.sam2_only and args.clip_only:
        parser.error("Cannot use both --sam2-only and --clip-only")

    try:
        # Export SAM2 unless --clip-only
        if not args.clip_only:
            Path(args.sam2_output).parent.mkdir(parents=True, exist_ok=True)
            export_sam2_onnx(args.sam2_output)

        # Export CLIP unless --sam2-only
        if not args.sam2_only:
            Path(args.clip_output).parent.mkdir(parents=True, exist_ok=True)
            export_clip_onnx(args.clip_output)

        print("\n" + "=" * 60)
        print("✓ All models exported successfully!")
        print("=" * 60)
        print("\nNext steps:")
        print("1. Build the Rust CLI: cargo build --release")
        print("2. Run detection: ./target/release/sam-detect-cli image.jpg")
        return 0

    except Exception as e:
        print(f"\n✗ Export failed: {e}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
