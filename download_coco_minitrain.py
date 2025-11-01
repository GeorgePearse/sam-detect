#!/usr/bin/env python3
"""Download COCO-minitrain dataset."""

import sys
from pathlib import Path

# Add parent to path for imports
sys.path.insert(0, str(Path(__file__).parent))

from sam_detect.download_datasets import download_coco_minitrain


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Download COCO-minitrain dataset")
    parser.add_argument(
        "--output-dir",
        default="./datasets/coco-minitrain",
        help="Output directory for dataset (default: ./datasets/coco-minitrain)",
    )

    args = parser.parse_args()

    output_dir = download_coco_minitrain(args.output_dir)
    print(f"\n✅ Dataset ready at: {output_dir}")
