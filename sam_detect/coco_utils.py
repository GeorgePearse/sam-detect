"""COCO dataset utilities for sam-detect evaluation."""

import json
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Tuple, Union

from PIL import Image


class COCODataset:
    """Load and iterate over COCO dataset images and annotations."""

    def __init__(
        self,
        images_dir: Union[str, Path],
        annotations_file: Union[str, Path],
        image_ids: Optional[List[int]] = None,
    ) -> None:
        """
        Initialize COCO dataset loader.

        Args:
            images_dir: Path to directory containing images
            annotations_file: Path to COCO JSON annotations file
            image_ids: Optional list of image IDs to load. If None, loads all.
        """
        self.images_dir = Path(images_dir)
        self.annotations_file = Path(annotations_file)

        # Load annotations
        with open(self.annotations_file) as f:
            self.data = json.load(f)

        # Build lookup tables
        self.images_by_id = {img["id"]: img for img in self.data.get("images", [])}
        self.categories = {cat["id"]: cat for cat in self.data.get("categories", [])}
        self.annotations_by_image = {}

        for ann in self.data.get("annotations", []):
            img_id = ann["image_id"]
            if img_id not in self.annotations_by_image:
                self.annotations_by_image[img_id] = []
            self.annotations_by_image[img_id].append(ann)

        # Filter image IDs if requested
        if image_ids is not None:
            self.image_ids = [i for i in image_ids if i in self.images_by_id]
        else:
            self.image_ids = list(self.images_by_id.keys())

    def __len__(self) -> int:
        """Return number of images in dataset."""
        return len(self.image_ids)

    def __iter__(self) -> Iterator[Tuple[int, Image.Image, List[Dict[str, Any]]]]:
        """Iterate over (image_id, image, annotations)."""
        for image_id in self.image_ids:
            image = self.load_image(image_id)
            annotations = self.annotations_by_image.get(image_id, [])
            yield image_id, image, annotations

    def load_image(self, image_id: int) -> Image.Image:
        """Load an image by ID."""
        img_meta = self.images_by_id[image_id]
        img_path = self.images_dir / img_meta["file_name"]

        if not img_path.exists():
            raise FileNotFoundError(f"Image not found: {img_path}")

        return Image.open(img_path).convert("RGB")

    def get_annotation_labels(self, annotations: List[Dict[str, Any]]) -> List[str]:
        """Get category names for annotations."""
        labels = []
        for ann in annotations:
            cat_id = ann.get("category_id")
            if cat_id in self.categories:
                labels.append(self.categories[cat_id]["name"])
        return labels

    def get_dataset_info(self) -> Dict[str, Any]:
        """Get dataset statistics."""
        return {
            "num_images": len(self.image_ids),
            "num_annotations": len(self.data.get("annotations", [])),
            "num_categories": len(self.categories),
            "categories": [cat["name"] for cat in self.categories.values()],
        }
