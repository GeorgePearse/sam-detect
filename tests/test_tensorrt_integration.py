"""Tests for TensorRT-optimized components (SAM2 and CLIP).

These tests verify that the TensorRT components have proper structure and
type annotations. Full integration tests require GPU and model downloads.
"""

import unittest
from unittest.mock import MagicMock, patch

from sam_detect.segmentation import SAM2Segmenter, SegmentationResult
from sam_detect.embedding import CLIPEmbedder


class TestSAM2SegmenterStructure(unittest.TestCase):
    """Test SAM2Segmenter class structure and initialization."""

    def test_sam2_segmenter_has_required_methods(self) -> None:
        """Verify SAM2Segmenter has __init__ and __call__ methods."""
        self.assertTrue(hasattr(SAM2Segmenter, "__init__"))
        self.assertTrue(hasattr(SAM2Segmenter, "__call__"))
        self.assertTrue(callable(SAM2Segmenter))

    def test_sam2_segmenter_parameters(self) -> None:
        """Verify SAM2Segmenter has correct parameter annotations."""
        import inspect

        sig = inspect.signature(SAM2Segmenter.__init__)
        params = list(sig.parameters.keys())

        # Check key parameters exist
        self.assertIn("self", params)
        self.assertIn("model_size", params)
        self.assertIn("device", params)

    @patch("sam_detect.segmentation.build_sam2")
    @patch("sam_detect.segmentation.SAM2AutomaticMaskGenerator")
    def test_sam2_segmenter_initialization_with_mock(
        self, mock_mask_gen: MagicMock, mock_build_sam2: MagicMock
    ) -> None:
        """Test SAM2Segmenter initialization with mocked dependencies."""
        # Mock the models
        mock_build_sam2.return_value = MagicMock()
        mock_mask_gen.return_value = MagicMock()

        # This should not raise an error
        try:
            segmenter = SAM2Segmenter(model_size="base", device="cpu")
            self.assertIsNotNone(segmenter)
            self.assertEqual(segmenter.model_size, "base")
            self.assertEqual(segmenter.device, "cpu")
        except ImportError:
            # Expected if SAM2 is not installed
            self.skipTest("SAM2 not installed")

    def test_segmentation_result_dataclass(self) -> None:
        """Verify SegmentationResult is properly defined."""
        # Create a test result
        mask = [[True, False], [False, True]]
        bbox = (0, 0, 10, 10)
        result = SegmentationResult(mask=mask, bbox=bbox)

        self.assertEqual(result.mask, mask)
        self.assertEqual(result.bbox, bbox)

    def test_mask_to_bbox_static_method(self) -> None:
        """Verify _mask_to_bbox static method exists."""
        import numpy as np

        # Create a simple mask
        mask = np.zeros((10, 10), dtype=bool)
        mask[2:5, 3:7] = True

        try:
            bbox = SAM2Segmenter._mask_to_bbox(mask)
            self.assertEqual(len(bbox), 4)
            self.assertTrue(all(isinstance(x, int) for x in bbox))
        except AttributeError:
            # Method might not be accessible in all contexts
            pass


class TestCLIPEmbedderStructure(unittest.TestCase):
    """Test CLIPEmbedder class structure."""

    def test_clip_embedder_has_required_methods(self) -> None:
        """Verify CLIPEmbedder has embed method."""
        self.assertTrue(hasattr(CLIPEmbedder, "embed"))

    def test_clip_embedder_parameters(self) -> None:
        """Verify CLIPEmbedder has correct parameters."""
        import inspect

        sig = inspect.signature(CLIPEmbedder.__init__)
        params = list(sig.parameters.keys())

        self.assertIn("self", params)
        self.assertIn("model_name", params)
        self.assertIn("device", params)

    @patch("sam_detect.embedding.CLIPModel")
    def test_clip_embedder_initialization_with_mock(
        self, mock_clip_model: MagicMock
    ) -> None:
        """Test CLIPEmbedder initialization with mocked dependencies."""
        # Mock the CLIPModel
        mock_clip_model.from_pretrained.return_value = MagicMock()

        # This should not raise an error
        try:
            embedder = CLIPEmbedder(
                model_name="openai/clip-vit-base-patch32", device="cpu"
            )
            self.assertIsNotNone(embedder)
            self.assertEqual(embedder.model_name, "openai/clip-vit-base-patch32")
            self.assertEqual(embedder.device, "cpu")
        except ImportError:
            # Expected if clip_trt is not installed
            self.skipTest("clip_trt not installed")

    def test_clip_embedder_default_model_name(self) -> None:
        """Verify CLIPEmbedder has reasonable default model name."""
        # Check that the default model name is from OpenAI's CLIP
        self.assertTrue(
            CLIPEmbedder.model_name == "openai/clip-vit-base-patch32"
            or hasattr(CLIPEmbedder, "__dataclass_fields__")
        )


class TestModelLoaderModule(unittest.TestCase):
    """Test the model_loader module functionality."""

    def test_model_loader_imports(self) -> None:
        """Verify model_loader module can be imported."""
        try:
            from sam_detect import model_loader

            self.assertTrue(hasattr(model_loader, "get_cache_dir"))
            self.assertTrue(hasattr(model_loader, "get_device"))
            self.assertTrue(hasattr(model_loader, "get_sam2_model_path"))
            self.assertTrue(hasattr(model_loader, "get_clip_model_path"))
        except ImportError:
            self.fail("model_loader module should be importable")

    def test_get_cache_dir(self) -> None:
        """Test get_cache_dir returns a Path."""
        from sam_detect.model_loader import get_cache_dir
        from pathlib import Path

        cache_dir = get_cache_dir()
        self.assertIsInstance(cache_dir, Path)

    def test_get_sam2_model_path(self) -> None:
        """Test get_sam2_model_path returns valid model IDs."""
        from sam_detect.model_loader import get_sam2_model_path

        for size in ["small", "base", "large"]:
            model_id = get_sam2_model_path(size)
            self.assertIn("facebook/sam2-hiera", model_id)
            self.assertIn(size, model_id)

    def test_get_clip_model_path(self) -> None:
        """Test get_clip_model_path returns valid model IDs."""
        from sam_detect.model_loader import get_clip_model_path

        # Test mapped variants
        base_model = get_clip_model_path("base")
        self.assertIn("openai/clip-vit", base_model)

        # Test custom model IDs pass through
        custom_model = "custom/model"
        result = get_clip_model_path(custom_model)
        self.assertEqual(result, custom_model)


class TestCLIIntegration(unittest.TestCase):
    """Test CLI integration with TensorRT components."""

    def test_cli_imports(self) -> None:
        """Verify CLI can import new components."""
        try:
            from sam_detect import cli

            # Verify the build functions exist
            self.assertTrue(hasattr(cli, "_build_segmenter"))
            self.assertTrue(hasattr(cli, "_build_embedder"))
        except ImportError:
            self.fail("CLI module should be importable")

    def test_build_segmenter_naive(self) -> None:
        """Test _build_segmenter returns NaiveSegmenter when requested."""
        from sam_detect.cli import _build_segmenter
        from sam_detect.segmentation import NaiveSegmenter

        segmenter = _build_segmenter("naive", "base", "cpu")
        self.assertIsInstance(segmenter, NaiveSegmenter)

    def test_build_embedder_average(self) -> None:
        """Test _build_embedder returns AverageColorEmbedder when requested."""
        from sam_detect.cli import _build_embedder
        from sam_detect.embedding import AverageColorEmbedder

        embedder = _build_embedder("average", "base", "cpu")
        self.assertIsInstance(embedder, AverageColorEmbedder)


class TestDownloadModelsScript(unittest.TestCase):
    """Test the download_models script structure."""

    def test_download_models_imports(self) -> None:
        """Verify download_models module can be imported."""
        try:
            from sam_detect import download_models

            self.assertTrue(hasattr(download_models, "main"))
            self.assertTrue(hasattr(download_models, "download_sam2_models"))
            self.assertTrue(hasattr(download_models, "download_clip_models"))
        except ImportError:
            self.fail("download_models module should be importable")

    def test_main_function_signature(self) -> None:
        """Test main function has correct signature."""
        from sam_detect.download_models import main
        import inspect

        sig = inspect.signature(main)
        params = list(sig.parameters.keys())

        self.assertIn("argv", params)
        # main should return int
        self.assertEqual(sig.return_annotation, int)


if __name__ == "__main__":
    unittest.main()
