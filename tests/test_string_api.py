"""Tests for the string-based API in sam-detect.

Comprehensive test suite for the new string-based SAMDetect API that allows
configuration using simple string identifiers instead of object instantiation.
"""

import unittest

import numpy as np
from PIL import Image

from sam_detect import SAMDetect, SEGMENTERS, EMBEDDERS, FADE_STRATEGIES, VECTOR_STORES
from sam_detect.model_registry import (
    build_segmenter,
    build_embedder,
    build_fade_strategy,
    build_vector_store,
)


class TestModelRegistryConstants(unittest.TestCase):
    """Test that model registry constants are properly exposed."""

    def test_segmenters_list_exists(self) -> None:
        """Verify SEGMENTERS constant is available."""
        self.assertIsInstance(SEGMENTERS, list)
        self.assertGreater(len(SEGMENTERS), 0)
        self.assertIn("naive", SEGMENTERS)
        self.assertIn("sam2-base", SEGMENTERS)

    def test_embedders_list_exists(self) -> None:
        """Verify EMBEDDERS constant is available."""
        self.assertIsInstance(EMBEDDERS, list)
        self.assertGreater(len(EMBEDDERS), 0)
        self.assertIn("average", EMBEDDERS)
        self.assertIn("clip-vit-base", EMBEDDERS)

    def test_fade_strategies_list_exists(self) -> None:
        """Verify FADE_STRATEGIES constant is available."""
        self.assertIsInstance(FADE_STRATEGIES, list)
        self.assertGreater(len(FADE_STRATEGIES), 0)
        self.assertIn("identity", FADE_STRATEGIES)
        self.assertIn("gaussian", FADE_STRATEGIES)

    def test_vector_stores_list_exists(self) -> None:
        """Verify VECTOR_STORES constant is available."""
        self.assertIsInstance(VECTOR_STORES, list)
        self.assertGreater(len(VECTOR_STORES), 0)
        self.assertIn("memory", VECTOR_STORES)
        self.assertIn("qdrant", VECTOR_STORES)


class TestSAMDetectDefaultInitialization(unittest.TestCase):
    """Test SAMDetect with default string-based configuration."""

    def test_default_initialization(self) -> None:
        """Test SAMDetect initializes with production-ready defaults."""
        try:
            detector = SAMDetect()
            self.assertIsNotNone(detector)
            # Check that defaults are set to production values
            self.assertEqual(detector.segmenter, "sam2-base")
            self.assertEqual(detector.embedder, "clip-vit-base")
            self.assertEqual(detector.fade_strategy, "gaussian")
            self.assertEqual(detector.vector_store, "memory")
            self.assertEqual(detector.device, "cuda")
        except ImportError:
            self.skipTest("SAM2 or CLIP dependencies not installed")

    def test_default_has_internal_objects(self) -> None:
        """Test that internal objects are built from defaults."""
        try:
            detector = SAMDetect()
            self.assertIsNotNone(detector._segmenter)
            self.assertIsNotNone(detector._embedder)
            self.assertIsNotNone(detector._fade_strategy)
            self.assertIsNotNone(detector._vector_store)
        except ImportError:
            self.skipTest("Required dependencies not installed")


class TestSAMDetectStringAPI(unittest.TestCase):
    """Test SAMDetect with various string configurations."""

    def test_segmenter_string_naive(self) -> None:
        """Test SAMDetect with naive segmenter string."""
        try:
            detector = SAMDetect(segmenter="naive")
            self.assertEqual(detector.segmenter, "naive")
            self.assertIsNotNone(detector._segmenter)
        except ImportError:
            self.skipTest("Required dependencies not installed")

    def test_embedder_string_average(self) -> None:
        """Test SAMDetect with average embedder string."""
        try:
            detector = SAMDetect(embedder="average")
            self.assertEqual(detector.embedder, "average")
            self.assertIsNotNone(detector._embedder)
        except ImportError:
            self.skipTest("Required dependencies not installed")

    def test_fade_strategy_string_identity(self) -> None:
        """Test SAMDetect with identity fade strategy."""
        try:
            detector = SAMDetect(fade_strategy="identity")
            self.assertEqual(detector.fade_strategy, "identity")
            self.assertIsNotNone(detector._fade_strategy)
        except ImportError:
            self.skipTest("Required dependencies not installed")

    def test_vector_store_string_memory(self) -> None:
        """Test SAMDetect with memory vector store."""
        try:
            detector = SAMDetect(vector_store="memory")
            self.assertEqual(detector.vector_store, "memory")
            self.assertIsNotNone(detector._vector_store)
        except ImportError:
            self.skipTest("Required dependencies not installed")

    def test_device_cpu(self) -> None:
        """Test SAMDetect with CPU device."""
        try:
            detector = SAMDetect(device="cpu")
            self.assertEqual(detector.device, "cpu")
        except ImportError:
            self.skipTest("SAM2 or CLIP dependencies not installed")

    def test_lightweight_cpu_configuration(self) -> None:
        """Test lightweight CPU-only configuration."""
        detector = SAMDetect(
            segmenter="naive",
            embedder="average",
            device="cpu",
        )
        self.assertEqual(detector.segmenter, "naive")
        self.assertEqual(detector.embedder, "average")
        self.assertEqual(detector.device, "cpu")
        self.assertIsNotNone(detector._segmenter)
        self.assertIsNotNone(detector._embedder)


class TestSAMDetectInvalidConfiguration(unittest.TestCase):
    """Test SAMDetect error handling with invalid strings."""

    def test_invalid_segmenter_raises_error(self) -> None:
        """Test that invalid segmenter string raises ValueError."""
        with self.assertRaises(ValueError):
            SAMDetect(segmenter="invalid_segmenter", embedder="average", device="cpu")

    def test_invalid_embedder_raises_error(self) -> None:
        """Test that invalid embedder string raises ValueError."""
        with self.assertRaises(ValueError):
            SAMDetect(segmenter="naive", embedder="invalid_embedder", device="cpu")

    def test_invalid_fade_strategy_raises_error(self) -> None:
        """Test that invalid fade strategy string raises ValueError."""
        with self.assertRaises(ValueError):
            SAMDetect(
                segmenter="naive",
                embedder="average",
                fade_strategy="invalid_fade",
                device="cpu",
            )

    def test_invalid_vector_store_raises_error(self) -> None:
        """Test that invalid vector store string raises ValueError."""
        with self.assertRaises(ValueError):
            SAMDetect(
                segmenter="naive",
                embedder="average",
                vector_store="invalid_store",
                device="cpu",
            )


class TestModelRegistryBuilders(unittest.TestCase):
    """Test individual model registry builder functions."""

    def test_build_segmenter_naive(self) -> None:
        """Test build_segmenter with naive option."""
        segmenter = build_segmenter("naive", "cpu")
        self.assertIsNotNone(segmenter)

    def test_build_embedder_average(self) -> None:
        """Test build_embedder with average option."""
        embedder = build_embedder("average", "cpu")
        self.assertIsNotNone(embedder)

    def test_build_fade_strategy_identity(self) -> None:
        """Test build_fade_strategy with identity option."""
        fade = build_fade_strategy("identity")
        self.assertIsNotNone(fade)

    def test_build_vector_store_memory(self) -> None:
        """Test build_vector_store with memory option."""
        store = build_vector_store("memory")
        self.assertIsNotNone(store)

    def test_build_embedder_with_huggingface_id(self) -> None:
        """Test build_embedder accepts HuggingFace model IDs."""
        try:
            embedder = build_embedder("openai/clip-vit-base-patch16", "cpu")
            self.assertIsNotNone(embedder)
        except ImportError:
            self.skipTest("CLIP dependencies not installed")

    def test_build_fade_strategy_gaussian_default(self) -> None:
        """Test build_fade_strategy with gaussian default parameters."""
        fade = build_fade_strategy("gaussian")
        self.assertIsNotNone(fade)

    def test_build_fade_strategy_gaussian_custom_sigma(self) -> None:
        """Test build_fade_strategy with custom sigma."""
        fade = build_fade_strategy("gaussian-sigma50")
        self.assertIsNotNone(fade)

    def test_build_fade_strategy_gaussian_custom_sigma_and_min(self) -> None:
        """Test build_fade_strategy with custom sigma and min_fade."""
        fade = build_fade_strategy("gaussian-sigma50-min0.2")
        self.assertIsNotNone(fade)


class TestSAMDetectDetectionFunctionality(unittest.TestCase):
    """Test actual detection functionality with string-based API."""

    def _create_test_image(self, size: int = 100) -> Image.Image:
        """Helper to create a test PIL image."""
        array = np.random.randint(0, 255, (size, size, 3), dtype=np.uint8)
        return Image.fromarray(array, mode="RGB")

    def test_detect_with_lightweight_models(self) -> None:
        """Test detect() works with lightweight CPU models."""
        detector = SAMDetect(
            segmenter="naive",
            embedder="average",
            device="cpu",
        )
        image = self._create_test_image()
        results = detector.detect(image)

        self.assertIsInstance(results, list)
        if results:  # NaiveSegmenter may or may not produce results
            result = results[0]
            self.assertIn("label", result)
            self.assertIn("confidence", result)
            self.assertIn("embedding", result)
            self.assertIn("result", result)

    def test_add_example_and_detect(self) -> None:
        """Test adding an example and then detecting."""
        detector = SAMDetect(
            segmenter="naive",
            embedder="average",
            device="cpu",
        )

        # Add an example
        image = self._create_test_image()
        embedding = detector.add_example(image, "test_label")
        self.assertIsNotNone(embedding)

        # Detect should still work
        results = detector.detect(image)
        self.assertIsInstance(results, list)

    def test_add_examples_batch(self) -> None:
        """Test adding multiple examples at once."""
        detector = SAMDetect(
            segmenter="naive",
            embedder="average",
            device="cpu",
        )

        images = [(self._create_test_image(), f"label_{i}") for i in range(3)]
        detector.add_examples(images)

        # Should not raise
        results = detector.detect(images[0][0])
        self.assertIsInstance(results, list)


class TestSAMDetectConfiguration(unittest.TestCase):
    """Test various SAMDetect configurations."""

    def test_with_qdrant_configuration(self) -> None:
        """Test SAMDetect with Qdrant configuration."""
        try:
            detector = SAMDetect(
                segmenter="naive",
                embedder="average",
                vector_store="qdrant",
                qdrant_url="http://localhost:6333",
                qdrant_collection="test_collection",
                device="cpu",
            )
            self.assertEqual(detector.vector_store, "qdrant")
            self.assertEqual(detector.qdrant_url, "http://localhost:6333")
            self.assertEqual(detector.qdrant_collection, "test_collection")
        except ImportError:
            self.skipTest("Required dependencies not installed")

    def test_with_top_k_parameter(self) -> None:
        """Test SAMDetect with custom top_k."""
        try:
            detector = SAMDetect(
                segmenter="naive",
                embedder="average",
                device="cpu",
                default_top_k=5,
            )
            self.assertEqual(detector.default_top_k, 5)
        except ImportError:
            self.skipTest("Required dependencies not installed")

    def test_all_segmenters_strings(self) -> None:
        """Test that all segmenter strings are valid."""
        for segmenter_str in SEGMENTERS:
            try:
                detector = SAMDetect(
                    segmenter=segmenter_str, embedder="average", device="cpu"
                )
                self.assertEqual(detector.segmenter, segmenter_str)
            except ImportError:
                # Skip if required dependencies not installed
                pass

    def test_all_embedders_strings(self) -> None:
        """Test that all embedder strings are valid."""
        for embedder_str in EMBEDDERS:
            try:
                detector = SAMDetect(
                    segmenter="naive", embedder=embedder_str, device="cpu"
                )
                self.assertEqual(detector.embedder, embedder_str)
            except ImportError:
                # Skip if required dependencies not installed
                pass

    def test_all_fade_strategies_strings(self) -> None:
        """Test that all fade strategy strings are valid."""
        for fade_str in FADE_STRATEGIES:
            try:
                detector = SAMDetect(
                    segmenter="naive",
                    embedder="average",
                    fade_strategy=fade_str,
                    device="cpu",
                )
                self.assertEqual(detector.fade_strategy, fade_str)
            except ImportError:
                pass

    def test_all_vector_stores_strings(self) -> None:
        """Test that all vector store strings are valid."""
        for store_str in VECTOR_STORES:
            try:
                detector = SAMDetect(
                    segmenter="naive",
                    embedder="average",
                    vector_store=store_str,
                    device="cpu",
                )
                self.assertEqual(detector.vector_store, store_str)
            except ImportError:
                pass


class TestStringAPIDiscoverability(unittest.TestCase):
    """Test that the string API is discoverable by users."""

    def test_segmenters_discoverable_from_import(self) -> None:
        """Test users can discover segmenters via import."""
        from sam_detect import SEGMENTERS

        self.assertIsInstance(SEGMENTERS, list)

    def test_embedders_discoverable_from_import(self) -> None:
        """Test users can discover embedders via import."""
        from sam_detect import EMBEDDERS

        self.assertIsInstance(EMBEDDERS, list)

    def test_fade_strategies_discoverable_from_import(self) -> None:
        """Test users can discover fade strategies via import."""
        from sam_detect import FADE_STRATEGIES

        self.assertIsInstance(FADE_STRATEGIES, list)

    def test_vector_stores_discoverable_from_import(self) -> None:
        """Test users can discover vector stores via import."""
        from sam_detect import VECTOR_STORES

        self.assertIsInstance(VECTOR_STORES, list)

    def test_example_usage_discoverable(self) -> None:
        """Test that common patterns work as documented."""
        # This is the pattern shown in README
        detector = SAMDetect(
            segmenter="naive",
            embedder="average",
            device="cpu",
        )
        self.assertIsNotNone(detector)

    def test_production_defaults_discoverable(self) -> None:
        """Test that production defaults are easily accessible."""
        # This should work without specifying parameters
        try:
            detector = SAMDetect()
            self.assertIsNotNone(detector)
        except ImportError:
            self.skipTest("Production dependencies not installed")


if __name__ == "__main__":
    unittest.main()
