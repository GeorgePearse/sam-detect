# sam-detect

A fast, adaptable instance segmentation system combining **SAM2** with **TensorRT optimization** and **few-shot learning** via **QDrant vector search**. Designed for real-time inference, batch processing, interactive annotation, and flexible research.

## Core Concept

Instead of training task-specific classifiers, sam-detect uses:

1. **Fast segmentation** (SAM2 + TensorRT) to isolate object instances
2. **Gaussian-faded context crops** to create information-rich embeddings without overwhelming noise
3. **Vector database retrieval** (QDrant) for few-shot classification via similarity search

This approach enables rapid adaptation to new object types with minimal examples while maintaining high inference speed.

## Architecture

```
Input Image
    ↓
[SAM2 + TensorRT] → Segmentation masks, bounding boxes
    ↓
[Gaussian Fading] → Context-aware crops (configurable fade strategy)
    ↓
[Feature Embedder] → Dense vectors (CLIP or similar)
    ↓
[QDrant Vector DB] → k-NN similarity search for classification
    ↓
Instance labels + confidence scores
```

### Key Components

#### 1. SAM2 with TensorRT
- **Fast segmentation**: SAM2 optimized with NVIDIA TensorRT for GPU acceleration
- **High quality**: Maintains SAM's impressive zero-shot segmentation capability
- **Target latency**: Segmentation in <50ms per frame (GPU-dependent)

#### 2. Gaussian Fading (Configurable)
Instead of hard crops, we apply **distance-based Gaussian fading** from the mask boundary:

```
Faded crop = image × exp(-(distance_from_mask / σ)²)
```

Where `σ` (sigma/width) is **configurable** to control context falloff:
- **Sharp focus** (σ=10): Tight focus on object, minimal context
- **Balanced** (σ=30): Moderate context for spatial relationships
- **Wide context** (σ=50+): Preserve surrounding visual information

This keeps the object sharp while gracefully fading distracting surroundings, improving embedding quality.

**Fade strategies are pluggable** - support for:
- Gaussian fade (default)
- Linear distance fade
- Exponential fade
- Custom user-defined strategies

#### 3. Feature Embeddings & Few-Shot Learning
- Embed faded crops using a frozen feature encoder (e.g., CLIP ViT)
- Store labeled examples as (embedding, label) pairs in QDrant
- Classify new instances via k-NN similarity search
- Minimal labeled data needed (few-shot learning)

#### 4. QDrant Vector Database
- **Fast retrieval**: Sub-millisecond k-NN search
- **Persistent storage**: Accumulate labeled examples over time
- **Flexible deployment**: Local or cloud instances
- **Scalable**: From hundreds to billions of vectors

## Features

- ✅ **Real-time inference** (<100ms per frame including segmentation + embedding + retrieval)
- ✅ **Batch processing** (few seconds per image for offline analysis)
- ✅ **Interactive annotation** (<1s feedback loop for human-in-the-loop labeling)
- ✅ **Research flexibility** (modular architecture, swappable components)
- ✅ **Configurable context fading** (adapt to different object types and scenarios)
- ✅ **Few-shot learning** (adapt to new classes with minimal labeled examples)
- ✅ **Production-ready** (GPU acceleration, vector indexing, distributed storage)

## Getting Started

### Prerequisites

- Python 3.10+
- CUDA 11.8+ (for TensorRT optimization)
- 8GB+ GPU VRAM (A10, RTX 3060+, or similar)
- Docker (optional, for QDrant)

### Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/sam-detect.git
cd sam-detect

# Install with uv (recommended)
uv pip install -e .

# Or with pip
pip install -e .
```

### CLI Quickstart

#### Basic Usage (with NaiveSegmenter)

```bash
# 1. Install the project (editable install recommended for dev work)
uv pip install -e .

# 2. Run a memory-backed detection with the default fade
sam-detect path/to/image.jpg

# 3. Add an example to prime the in-memory store, then detect again
sam-detect path/to/image.jpg --label widget

# 4. Optional: start Qdrant and enable retrieval + Gaussian fading
docker run -p 6333:6333 qdrant/qdrant:latest
sam-detect path/to/image.jpg \
  --store qdrant --qdrant-url http://localhost:6333 \
  --fade gaussian --sigma 30 --min-fade 0.05 \
  --top-k 5 --json
```

#### Production Usage (with SAM2 + CLIP + TensorRT)

```bash
# 1. Download and cache models (one-time setup)
pip install 'sam-detect[tensorrt]'
python -m sam_detect.download_models

# 2. Run with SAM2 segmentation and CLIP embeddings
sam-detect path/to/image.jpg \
  --segmenter sam2 --sam-model base \
  --embedder clip --device cuda

# 3. Or with Qdrant for persistent storage
sam-detect path/to/image.jpg \
  --segmenter sam2 --sam-model base \
  --embedder clip --device cuda \
  --store qdrant --qdrant-url http://localhost:6333 \
  --fade gaussian --sigma 30 --top-k 5 --json
```

**Key CLI flags:**
- `--segmenter {naive|sam2}` - Segmentation backend (naive for testing, sam2 for production)
- `--sam-model {small|base|large}` - SAM2 model size (base balances speed/accuracy)
- `--embedder {average|clip}` - Embedding strategy (clip requires TensorRT install)
- `--device {cuda|cpu}` - Device for inference (CUDA required for TensorRT)
- `--store {memory|qdrant}` - Vector store backend
- `--fade {identity|gaussian}`, `--sigma`, `--min-fade` - Context fading tuning
- `--top-k` - Number of nearest neighbours to return
- `--json` - Emit JSON instead of text

### Quick Start (Python API)

#### 1. Install with TensorRT Support

```bash
# Install SAM2 + CLIP + TensorRT dependencies
pip install 'sam-detect[tensorrt]'

# Or install individual components
pip install -e .  # Base installation
pip install segment-anything-2  # SAM2
pip install clip-trt  # CLIP with TensorRT
pip install torch-tensorrt  # TensorRT compiler
```

#### 2. Download and Cache Models

```bash
# One-time setup: download SAM2 Base and CLIP Base models
python -m sam_detect.download_models

# Download specific variants:
python -m sam_detect.download_models --sam-model large
python -m sam_detect.download_models --clip-model openai/clip-vit-large-patch14
```

This downloads and caches:
- SAM2 model weights (specified variant)
- CLIP model weights (specified variant)
- TensorRT compiled engines (cached for reuse)

#### 3. Optional: Start QDrant for Persistent Storage

```bash
# Local QDrant (Docker)
docker run -p 6333:6333 qdrant/qdrant:latest

# Or use Qdrant Cloud (managed, requires API key)
export QDRANT_API_KEY=your_api_key
export QDRANT_URL=https://your-instance.qdrant.io
```

#### 4. Initialize with SAM2 + CLIP

```python
from sam_detect import SAMDetect
from sam_detect.segmentation import SAM2Segmenter
from sam_detect.embedding import CLIPEmbedder
from sam_detect.fading import GaussianFade

# Create detector with SAM2 segmentation and CLIP embeddings
detector = SAMDetect(
    segmenter=SAM2Segmenter(model_size="base", device="cuda"),
    embedder=CLIPEmbedder(model_name="openai/clip-vit-base-patch32", device="cuda"),
    fade_strategy=GaussianFade(sigma=30, min_fade=0.1),
)

# With Qdrant for persistent storage
from sam_detect.vector_store import QdrantVectorStore

detector = SAMDetect(
    segmenter=SAM2Segmenter(model_size="base", device="cuda"),
    embedder=CLIPEmbedder(device="cuda"),
    fade_strategy=GaussianFade(sigma=30),
    vector_store=QdrantVectorStore(
        url="http://localhost:6333",
        collection_name="sam_detect"
    ),
    default_top_k=5,
)
```

## Usage Examples

### Real-Time Video Processing

```python
import cv2
from sam_detect import SAMDetect

detector = SAMDetect(device="cuda")

# First, add some labeled examples to QDrant
detector.add_examples_from_file(
    image_path="person.jpg",
    masks="auto",  # Auto-segment with SAM2
    label="person"
)

# Process video stream
cap = cv2.VideoCapture(0)  # Webcam

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Segment + embed + classify in one call
    results = detector.detect(frame)

    # results = [
    #     {"mask": ndarray, "label": "person", "confidence": 0.94},
    #     {"mask": ndarray, "label": "person", "confidence": 0.87},
    # ]

    # Visualize
    viz = detector.visualize(frame, results)
    cv2.imshow("sam-detect", viz)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
```

### Batch Processing with Adaptive Learning

```python
from sam_detect import SAMDetect
from pathlib import Path

detector = SAMDetect(device="cuda")

# Process images and progressively add labeled examples
image_dir = Path("datasets/objects")

for image_path in image_dir.glob("*.jpg"):
    results = detector.detect(str(image_path))

    # Show results to user (or use model predictions if confident)
    for i, result in enumerate(results):
        # Save crop for later labeling
        crop = detector.get_crop(image_path, result["mask"])
        crop.save(f"crops/{image_path.stem}_{i}.png")

        # If confident prediction, add to database
        if result["confidence"] > 0.85:
            detector.add_example(
                image=crop,
                label=result["label"]
            )

print(f"Added {len(results)} examples to QDrant")
```

### Interactive Annotation Tool

```python
from sam_detect import SAMDetect, AnnotationUI

detector = SAMDetect(device="cuda")
annotation_ui = AnnotationUI(detector)

# Launch interactive web UI for rapid annotation
# - Shows SAM2 segmentation
# - Stores labeled examples in QDrant
# - Provides instant feedback (1s per image)
# - Suggests labels based on similarity search
annotation_ui.run(
    image_directory="unlabeled_data/",
    port=8000
)

# Visit http://localhost:8000 to start annotating
```

### Research & Experimentation

```python
from sam_detect import SAMDetect
from sam_detect.fading import GaussianFade, LinearFade, ExponentialFade

detector = SAMDetect(device="cuda")

# Experiment with different fading strategies
for fade_strategy in [GaussianFade(sigma=20),
                       LinearFade(max_distance=100),
                       ExponentialFade(decay=0.05)]:

    detector.set_fade_strategy(fade_strategy)

    # Evaluate on benchmark
    metrics = detector.evaluate(
        dataset_path="benchmark/",
        metrics=["accuracy", "inference_time", "embedding_quality"]
    )

    print(f"{fade_strategy.__class__.__name__}: {metrics}")
```

## Configuration

### Fading Strategies

Configure how context fades from the segmented object:

```python
from sam_detect import SAMDetect
from sam_detect.fading import GaussianFade, LinearFade

# Gaussian fade (smooth, recommended)
detector = SAMDetect(
    fade_config=GaussianFade(
        sigma=30,           # Width of Gaussian (pixels)
        min_fade=0.05,      # Avoid complete darkness
    )
)

# Linear fade (simple, predictable)
detector = SAMDetect(
    fade_config=LinearFade(
        max_distance=100,   # Distance at which fade = 0
        min_fade=0.05,
    )
)

# Exponential fade (aggressive context suppression)
detector = SAMDetect(
    fade_config=ExponentialFade(
        decay=0.05,         # Decay rate
        max_distance=200,
    )
)

# Custom fade strategy
class CustomFade:
    def apply(self, image, mask, distance_map):
        # distance_map: distance from mask boundary (pixels)
        # Return faded image
        pass

detector = SAMDetect(fade_config=CustomFade())
```

### Performance Tuning

```python
detector = SAMDetect(
    sam_model="sam2_hiera_large",  # Use "small" or "base" for speed
    device="cuda",
    tensorrt_enabled=True,          # Enable TensorRT optimization
    embedding_cache_size=10000,     # Cache embeddings for speed
    qdrant_batch_size=32,           # k-NN batch size
    embedding_model="clip-vit-b",   # Choose embedder
)
```

## Performance Benchmarks

Approximate latencies (GPU: RTX 4090):

| Stage | Time | Notes |
|-------|------|-------|
| SAM2 Segmentation | 45ms | Per frame, includes mask post-processing |
| Gaussian Fading | 5ms | Per object |
| Feature Embedding | 10ms | Per object, batched |
| QDrant k-NN Search | <1ms | Per object, 1000-vector index |
| **Total (5 objects)** | **~150ms** | Parallelizable stages |

For real-time video: 6-7 FPS per GPU at 1080p with 5-10 objects per frame.

**See also:** [Performance Analysis](docs/performance.md) for speed vs accuracy trade-off visualization and configuration recommendations.

## Roadmap

- [ ] **Phase 1 (MVP)**: Core segmentation + fading + QDrant integration
- [ ] **Phase 2 (Speed)**: TensorRT optimization, batch inference, embedding caching
- [ ] **Phase 3 (UX)**: Interactive annotation web UI, model analysis tools
- [ ] **Phase 4 (Production)**: Distributed QDrant, model serving (TorchServe/ONNX), monitoring
- [ ] **Phase 5 (Research)**: Multi-modal embeddings, active learning, online adaptation

## TODO: Metrics Dashboard

Similar to [rust-analyzer metrics](https://rust-analyzer.github.io/metrics/), we should build:

- **Performance tracking dashboard**: Track inference latency, embedding quality, and detection accuracy over time
- **Continuous benchmarking**: Automated evaluation on COCO/custom datasets after each commit
- **Metrics visualization**: Charts showing:
  - Average inference time per component (segmentation, embedding, k-NN search)
  - Detection accuracy/mAP across different fading strategies
  - Embedding quality metrics (embedding similarity, clustering)
  - Memory usage and throughput
  - Few-shot learning convergence curves
- **Regression detection**: Alert on performance degradation
- **Multi-backend comparison**: Compare performance across embedders (average, CLIP) and backends (memory, QDrant)
- **Hardware tracking**: Benchmark across different GPUs (A10, RTX 3060, RTX 4090, etc.)

This would enable transparent performance monitoring and help identify optimization opportunities.

## Contributing

Contributions welcome! Areas of interest:

- **Optimization**: Faster SAM variants, TensorRT tuning, GPU batch processing
- **Fading strategies**: Novel distance-based fading methods
- **Embedding models**: Experimental feature encoders
- **Evaluation**: Benchmark suite for few-shot learning scenarios
- **Applications**: Real-world use case implementations

## License

MIT License - see LICENSE file for details

## Acknowledgments

**Core Models & Libraries:**
- [SAM2](https://github.com/facebookresearch/segment-anything-2) - Meta AI's Segment Anything Model 2 for zero-shot instance segmentation
- [CLIP](https://github.com/openai/CLIP) - OpenAI's vision-language model for semantic understanding
- [QDrant](https://qdrant.tech/) - High-performance vector database for similarity search

**TensorRT Optimization:**
- [PyTorch TensorRT SAM2 Tutorial](https://docs.pytorch.org/TensorRT/tutorials/_rendered_examples/dynamo/torch_export_sam2.html) - Official guide for SAM2 TensorRT optimization using torch.export and dynamo backend
- [clip_trt](https://github.com/dusty-nv/clip_trt) - Optimized CLIP inference with TensorRT by dusty-nv
- [TensorRT](https://developer.nvidia.com/tensorrt) - NVIDIA's inference optimization framework

**Research Inspiration:**
- Few-shot learning and instance segmentation research
- Vision-language models for zero-shot classification
- Efficient inference optimization techniques
