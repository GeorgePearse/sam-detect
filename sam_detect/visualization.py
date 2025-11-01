"""Visualization utilities for detection results."""

from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from PIL import Image, ImageDraw, ImageFont


def draw_segmentation_mask(
    image: Image.Image,
    mask: List[List[int]],
    color: Tuple[int, int, int] = (0, 255, 0),
    alpha: float = 0.3,
) -> Image.Image:
    """
    Draw a segmentation mask on an image.

    Args:
        image: PIL Image to draw on
        mask: 2D list representing mask (0 or 1)
        color: RGB color for mask
        alpha: Transparency of mask overlay

    Returns:
        New image with mask drawn
    """
    img_copy = image.copy()

    if not mask or not mask[0]:
        return img_copy

    # Create mask image
    mask_array = np.array(mask, dtype=np.uint8) * 255
    mask_img = Image.fromarray(mask_array, mode="L")

    # Resize mask to match image if needed
    if mask_img.size != img_copy.size:
        mask_img = mask_img.resize(img_copy.size, Image.NEAREST)

    # Create colored overlay
    overlay = Image.new("RGB", img_copy.size, color)

    # Blend using mask
    img_copy.paste(overlay, (0, 0), mask_img)

    return img_copy


def draw_bounding_box(
    image: Image.Image,
    bbox: Tuple[float, float, float, float],
    label: str = "",
    color: Tuple[int, int, int] = (0, 255, 0),
    width: int = 2,
) -> Image.Image:
    """
    Draw a bounding box on an image.

    Args:
        image: PIL Image to draw on
        bbox: Bounding box as (x_min, y_min, x_max, y_max)
        label: Text label to display
        color: RGB color for box
        width: Box line width

    Returns:
        New image with bounding box drawn
    """
    img_copy = image.copy()
    draw = ImageDraw.Draw(img_copy)

    # Draw box
    draw.rectangle(bbox, outline=color, width=width)

    # Draw label if provided
    if label:
        try:
            font = ImageFont.load_default()
        except Exception:
            font = None

        # Get text bounding box for background
        text_bbox = draw.textbbox((bbox[0], bbox[1]), label, font=font)
        text_bg_padding = 2

        # Draw text background
        bg_bbox = (
            text_bbox[0] - text_bg_padding,
            text_bbox[1] - text_bg_padding,
            text_bbox[2] + text_bg_padding,
            text_bbox[3] + text_bg_padding,
        )
        draw.rectangle(bg_bbox, fill=color)

        # Draw text
        draw.text((bbox[0], bbox[1]), label, fill=(0, 0, 0), font=font)

    return img_copy


def visualize_detection_result(
    image: Image.Image,
    detections: List[Dict[str, Any]],
    show_confidence: bool = True,
    max_detections: Optional[int] = None,
) -> Image.Image:
    """
    Visualize detection results on an image.

    Args:
        image: PIL Image
        detections: List of detection results from SAMDetect.detect()
        show_confidence: Whether to include confidence in labels
        max_detections: Maximum number of detections to visualize

    Returns:
        Image with detections drawn
    """
    colors = [
        (255, 0, 0),  # Red
        (0, 255, 0),  # Green
        (0, 0, 255),  # Blue
        (255, 255, 0),  # Yellow
        (255, 0, 255),  # Magenta
        (0, 255, 255),  # Cyan
    ]

    result = image.copy()

    # Limit number of detections if specified
    dets = detections[:max_detections] if max_detections else detections

    for idx, detection in enumerate(dets):
        color = colors[idx % len(colors)]

        # Get label with optional confidence
        label = detection.get("label", "unknown")
        if show_confidence:
            confidence = detection.get("confidence", 0.0)
            label = f"{label} ({confidence:.2f})"

        # Draw bounding box if available
        result_obj = detection.get("result")
        if result_obj and hasattr(result_obj, "bbox"):
            bbox = result_obj.bbox
            result = draw_bounding_box(result, bbox, label, color=color)

    return result


def create_grid_visualization(
    images_with_detections: List[Tuple[Image.Image, List[Dict[str, Any]]]],
    grid_size: Tuple[int, int] = (2, 2),
) -> Image.Image:
    """
    Create a grid visualization of multiple detection results.

    Args:
        images_with_detections: List of (image, detections) tuples
        grid_size: Grid layout as (rows, cols)

    Returns:
        Single image with all results in grid
    """
    rows, cols = grid_size
    num_images = len(images_with_detections)
    num_cells = rows * cols

    if num_images > num_cells:
        images_with_detections = images_with_detections[:num_cells]

    # Assume all images same size, use first as reference
    if not images_with_detections:
        return Image.new("RGB", (100, 100), color="white")

    sample_img = images_with_detections[0][0]
    cell_width, cell_height = sample_img.size

    # Create grid image
    grid_width = cols * cell_width
    grid_height = rows * cell_height
    grid_img = Image.new("RGB", (grid_width, grid_height), color="white")

    # Place images in grid
    for idx, (image, detections) in enumerate(images_with_detections):
        row = idx // cols
        col = idx % cols

        # Visualize this image
        vis_image = visualize_detection_result(image, detections, max_detections=5)

        # Resize if needed
        if vis_image.size != (cell_width, cell_height):
            vis_image = vis_image.resize((cell_width, cell_height), Image.LANCZOS)

        # Paste into grid
        x = col * cell_width
        y = row * cell_height
        grid_img.paste(vis_image, (x, y))

    return grid_img
