"""Minimal scaffolding for the sam-detect package."""

from .pipeline import SAMDetect
from .model_registry import SEGMENTERS, EMBEDDERS, FADE_STRATEGIES, VECTOR_STORES

__all__ = [
    "SAMDetect",
    "SEGMENTERS",
    "EMBEDDERS",
    "FADE_STRATEGIES",
    "VECTOR_STORES",
]
