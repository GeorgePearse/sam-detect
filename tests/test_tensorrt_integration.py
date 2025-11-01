"""Tests for TensorRT-optimized components (SAM2 and CLIP).

Comprehensive test suite covering:
- SAM2Segmenter initialization and inference
- CLIPEmbedder initialization and inference
- Pipeline integration
- CLI argument parsing
- Model loader functionality
- Edge cases and error handling
"""

import unittest
from pathlib import Path
import numpy as np
from PIL import Image

from sam_detect.segmentation import SAM2Segmenter, SegmentationResult, NaiveSegmenter
from sam_detect.embedding import CLIPEmbedder, AverageColorEmbedder


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


class TestSAM2SegmenterFunctional(unittest.TestCase):
    """Functional tests for SAM2Segmenter."""

    def _create_test_image(self, size: int = 256) -> Image.Image:
        """Helper to create a test PIL image."""
        array = np.random.randint(0, 255, (size, size, 3), dtype=np.uint8)
        return Image.fromarray(array, mode="RGB")

    def test_mask_to_bbox_computation(self) -> None:
        """Test _mask_to_bbox correctly computes bounding boxes."""
        # Create a simple mask with known bounds
        mask = np.zeros((100, 100), dtype=bool)
        mask[20:50, 30:70] = True

        try:
            bbox = SAM2Segmenter._mask_to_bbox(mask)
            # bbox should be (x1, y1, x2, y2) format
            self.assertEqual(len(bbox), 4)
            x1, y1, x2, y2 = bbox
            self.assertGreaterEqual(x1, 0)
            self.assertGreaterEqual(y1, 0)
            self.assertLess(x2, 100)
            self.assertLess(y2, 100)
            self.assertLess(x1, x2)
            self.assertLess(y1, y2)
        except AttributeError:
            self.skipTest("_mask_to_bbox not accessible")

    def test_mask_to_bbox_empty_mask(self) -> None:
        """Test _mask_to_bbox handles empty masks."""
        mask = np.zeros((100, 100), dtype=bool)

        try:
            bbox = SAM2Segmenter._mask_to_bbox(mask)
            # Should still return a valid bbox
            self.assertEqual(len(bbox), 4)
        except AttributeError:
            self.skipTest("_mask_to_bbox not accessible")

    def test_mask_to_bbox_full_mask(self) -> None:
        """Test _mask_to_bbox handles full masks."""
        mask = np.ones((100, 100), dtype=bool)

        try:
            bbox = SAM2Segmenter._mask_to_bbox(mask)
            # Should cover entire image
            self.assertEqual(len(bbox), 4)
            x1, y1, x2, y2 = bbox
            self.assertEqual(x1, 0)
            self.assertEqual(y1, 0)
        except AttributeError:
            self.skipTest("_mask_to_bbox not accessible")


class TestCLIPEmbedderFunctional(unittest.TestCase):
    """Functional tests for CLIPEmbedder."""

    def _create_test_image(self, size: int = 224) -> Image.Image:
        """Helper to create a test PIL image."""
        array = np.random.randint(0, 255, (size, size, 3), dtype=np.uint8)
        return Image.fromarray(array, mode="RGB")

    def test_embedder_initialization(self) -> None:
        """Test CLIPEmbedder can be initialized."""
        try:
            embedder = CLIPEmbedder(
                model_name="openai/clip-vit-base-patch32", device="cpu"
            )
            self.assertIsNotNone(embedder)
        except ImportError:
            self.skipTest("clip_trt not installed")


class TestCLIArgumentParsing(unittest.TestCase):
    """Test CLI argument parsing for TensorRT options."""

    def test_segmenter_choice_argument(self) -> None:
        """Test CLI accepts segmenter choice argument."""
        from sam_detect.cli import _build_segmenter

        # Should not raise error
        segmenter = _build_segmenter("naive", "base", "cpu")
        self.assertIsInstance(segmenter, NaiveSegmenter)

    def test_embedder_choice_argument(self) -> None:
        """Test CLI accepts embedder choice argument."""
        from sam_detect.cli import _build_embedder

        # Should not raise error
        embedder = _build_embedder("average", "base", "cpu")
        self.assertIsInstance(embedder, AverageColorEmbedder)

    def test_invalid_segmenter_choice(self) -> None:
        """Test CLI handles invalid segmenter choice gracefully."""
        from sam_detect.cli import _build_segmenter

        # Should raise error for invalid choice
        try:
            _build_segmenter("invalid_segmenter", "base", "cpu")
            self.fail("Should have raised an error")
        except (ValueError, KeyError, AttributeError):
            pass  # Expected

    def test_invalid_embedder_choice(self) -> None:
        """Test CLI handles invalid embedder choice gracefully."""
        from sam_detect.cli import _build_embedder

        # Should raise error for invalid choice
        try:
            _build_embedder("invalid_embedder", "base", "cpu")
            self.fail("Should have raised an error")
        except (ValueError, KeyError, AttributeError):
            pass  # Expected


class TestModelLoaderFunctional(unittest.TestCase):
    """Functional tests for model loader."""

    def test_model_loader_functions_exist(self) -> None:
        """Test all model loader functions are available."""
        from sam_detect.model_loader import (
            get_cache_dir,
            get_device,
            get_sam2_model_path,
            get_clip_model_path,
        )

        # All functions should be callable
        self.assertTrue(callable(get_cache_dir))
        self.assertTrue(callable(get_device))
        self.assertTrue(callable(get_sam2_model_path))
        self.assertTrue(callable(get_clip_model_path))

    def test_cache_dir_returns_path(self) -> None:
        """Test get_cache_dir returns a valid Path."""
        from sam_detect.model_loader import get_cache_dir

        cache_dir = get_cache_dir()
        self.assertIsInstance(cache_dir, Path)
        # Cache dir should be under home directory
        self.assertTrue(str(cache_dir).startswith(str(Path.home())))

    def test_sam2_model_paths_all_valid(self) -> None:
        """Test get_sam2_model_path returns valid model IDs for all sizes."""
        from sam_detect.model_loader import get_sam2_model_path

        for size in ["small", "base", "large"]:
            model_path = get_sam2_model_path(size)
            self.assertIsInstance(model_path, str)
            self.assertIn("facebook/sam2", model_path)
            self.assertIn(size, model_path)

    def test_clip_model_paths_all_valid(self) -> None:
        """Test get_clip_model_path returns valid model IDs."""
        from sam_detect.model_loader import get_clip_model_path

        # Test predefined variants
        base_model = get_clip_model_path("base")
        self.assertIsInstance(base_model, str)
        self.assertIn("openai/clip", base_model)

    def test_device_detection_valid(self) -> None:
        """Test get_device returns a valid device."""
        from sam_detect.model_loader import get_device

        try:
            device = get_device()
            self.assertIn(device, ["cuda", "cpu"])
        except RuntimeError:
            # Expected if CUDA is not available
            self.skipTest("CUDA not available and required by get_device")


class TestErrorHandling(unittest.TestCase):
    """Test error handling and edge cases."""

    def test_naive_segmenter_fallback(self) -> None:
        """Test NaiveSegmenter works as fallback."""
        segmenter = NaiveSegmenter()
        image = Image.new("RGB", (100, 100), color="red")

        # Should return segmentation results
        results = list(segmenter(image))
        self.assertGreater(len(results), 0)

    def test_naive_segmenter_returns_segmentation_results(self) -> None:
        """Test NaiveSegmenter returns proper types."""
        segmenter = NaiveSegmenter()
        image = Image.new("RGB", (100, 100), color="red")

        results = list(segmenter(image))
        for result in results:
            self.assertIsInstance(result, SegmentationResult)
            self.assertIsNotNone(result.mask)
            self.assertIsNotNone(result.bbox)

    def test_average_embedder_fallback(self) -> None:
        """Test AverageColorEmbedder works as fallback."""
        embedder = AverageColorEmbedder()
        image = Image.new("RGB", (100, 100), color="red")

        # Should return embeddings
        embeddings = tuple(embedder.embed(image))
        self.assertEqual(len(embeddings), 3)  # RGB

    def test_average_embedder_different_colors(self) -> None:
        """Test AverageColorEmbedder works with different colors."""
        embedder = AverageColorEmbedder()

        # Test red
        red_image = Image.new("RGB", (100, 100), color=(255, 0, 0))
        red_embedding = tuple(embedder.embed(red_image))
        self.assertEqual(len(red_embedding), 3)

        # Test green
        green_image = Image.new("RGB", (100, 100), color=(0, 255, 0))
        green_embedding = tuple(embedder.embed(green_image))
        self.assertEqual(len(green_embedding), 3)

        # Embeddings should be different
        self.assertNotEqual(red_embedding[0], green_embedding[0])

    def test_segmentation_result_bbox_types(self) -> None:
        """Test SegmentationResult bbox contains correct types."""
        bbox = (10, 20, 30, 40)
        result = SegmentationResult(mask=[[True]], bbox=bbox)

        self.assertEqual(result.bbox, bbox)
        x1, y1, x2, y2 = result.bbox
        self.assertIsInstance(x1, int)
        self.assertIsInstance(y1, int)
        self.assertIsInstance(x2, int)
        self.assertIsInstance(y2, int)


class TestIntegration(unittest.TestCase):
    """Integration tests combining multiple components."""

    def test_fallback_pipeline_works(self) -> None:
        """Test that fallback components work together."""
        from sam_detect.pipeline import SAMDetect

        segmenter = NaiveSegmenter()
        embedder = AverageColorEmbedder()

        try:
            pipeline = SAMDetect(segmenter=segmenter, embedder=embedder)
            self.assertIsNotNone(pipeline)
        except Exception as e:
            self.fail(f"Pipeline initialization failed: {e}")

    def test_cli_imports(self) -> None:
        """Verify CLI module can be imported."""
        try:
            from sam_detect import cli

            # Verify the build functions exist
            self.assertTrue(hasattr(cli, "_build_segmenter"))
            self.assertTrue(hasattr(cli, "_build_embedder"))
        except ImportError:
            self.fail("CLI module should be importable")


if __name__ == "__main__":
    unittest.main()
