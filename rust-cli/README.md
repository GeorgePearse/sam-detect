# sam-detect Rust CLI

High-performance Rust implementation of sam-detect using ONNX Runtime with TensorRT acceleration.

## Overview

This is a **separate, complementary implementation** of the sam-detect pipeline in Rust, designed for production deployments requiring:
- **Zero garbage collection** - predictable latency
- **Single static binary** - easy deployment
- **Native TensorRT support** - direct ONNX Runtime integration
- **Async vector search** - efficient Qdrant queries

## Architecture

```
Image Input
    ↓
[Rust CLI] ← Load ONNX models
    ├→ [SAM2 via ort] → Segmentation masks
    ├→ [CLIP via ort] → Image embeddings
    └→ [Qdrant client] → k-NN similarity search
         ↓
Output: Labeled detections
```

## Prerequisites

- **Rust 1.75+** ([Install](https://rustup.rs/))
- **CUDA 12.x** (for TensorRT)
- **libssl-dev** (for qdrant-client)
- **Qdrant server** (Docker: `docker run -p 6333:6333 qdrant/qdrant:latest`)

### Linux Dependencies

```bash
# Ubuntu/Debian
sudo apt-get install -y libssl-dev pkg-config

# macOS
brew install openssl pkg-config
```

## Building

```bash
# Clone the repo (if not already in sam-detect)
cd sam-detect/rust-cli

# Build in release mode (optimized)
cargo build --release

# Binary location: target/release/sam-detect-cli
```

## Model Export

First, export PyTorch models to ONNX format:

```bash
# Install Python dependencies (in sam-detect root)
pip install 'sam-detect[tensorrt]'

# Export models
cd rust-cli
python scripts/export_models.py

# Models will be created:
# - models/sam2_base.onnx (~500 MB)
# - models/clip_vit_base.onnx (~100 MB)
```

## Usage

### Basic Detection

```bash
./target/release/sam-detect-cli image.jpg
```

### With Qdrant Integration

```bash
# 1. Start Qdrant
docker run -p 6333:6333 qdrant/qdrant:latest

# 2. Run detection
./target/release/sam-detect-cli image.jpg \
  --qdrant-url http://localhost:6333 \
  --collection-name sam_detect

# 3. Add training examples
./target/release/sam-detect-cli example1.jpg \
  --label "person" \
  --qdrant-url http://localhost:6333

# 4. Detect with retrieval
./target/release/sam-detect-cli test.jpg \
  --qdrant-url http://localhost:6333 \
  --top-k 5
```

### JSON Output

```bash
./target/release/sam-detect-cli image.jpg --format json
```

### Memory-Only Mode (No Qdrant)

```bash
./target/release/sam-detect-cli image.jpg --skip-qdrant
```

### Verbose Logging

```bash
# Debug: -v
# Trace: -vv
./target/release/sam-detect-cli image.jpg -vv
```

## CLI Options

```
sam-detect-cli <IMAGE_PATH> [OPTIONS]

ARGS:
  <IMAGE_PATH>...  Image file(s) to process

OPTIONS:
  --sam2-model <PATH>
      SAM2 ONNX model path [default: models/sam2_base.onnx]

  --clip-model <PATH>
      CLIP ONNX model path [default: models/clip_vit_base.onnx]

  --qdrant-url <URL>
      Qdrant server URL [default: http://localhost:6333]

  --collection-name <NAME>
      Qdrant collection name [default: sam_detect]

  --top-k <K>
      Number of nearest neighbors to return [default: 5]

  --format <FORMAT>
      Output format (json or text) [default: text]

  --label <LABEL>
      Add this label to the database before detection

  --skip-qdrant
      Skip Qdrant connection (memory-only mode)

  -v, --verbose
      Verbose logging (can be repeated: -v, -vv, -vvv)

  -h, --help
      Print help

  -V, --version
      Print version
```

## Performance

Expected performance on RTX 4090:

| Operation | Time |
|-----------|------|
| SAM2 Base segmentation | 50-100ms |
| CLIP ViT-Base embedding | 10-20ms |
| Qdrant k-NN search | <1ms |
| **Total (per image)** | **~100-150ms** |

For comparison:
- **Python CLI** with TensorRT: 150-250ms (includes Python overhead)
- **Rust CLI** with TensorRT: 100-150ms (native execution)

## Examples

### Real-World Usage

```bash
# Detect objects in webcam stream (Unix pipeline)
ls images/*.jpg | while read img; do
  ./target/release/sam-detect-cli "$img" --format json | jq '.detections'
done

# Batch processing with labels
./target/release/sam-detect-cli \
  dataset/*.jpg \
  --label "person" \
  --qdrant-url http://localhost:6333

# Comparison with Python CLI
./target/release/sam-detect-cli test.jpg --format json > rust_output.json
python -m sam_detect.cli test.jpg --json > python_output.json
diff <(jq .detections rust_output.json | sort) \
     <(jq .detections python_output.json | sort)
```

## Development

### Running Tests

```bash
# All tests
cargo test

# With logging
RUST_LOG=debug cargo test -- --nocapture

# Specific test
cargo test models::tests::test_preprocess_for_sam2
```

### Benchmarks

```bash
cargo bench --bench inference
```

### Code Quality

```bash
# Format
cargo fmt

# Lint
cargo clippy --all-targets --all-features

# Type check
cargo check
```

## Troubleshooting

### "Failed to load ONNX model"

Ensure models are exported:
```bash
python scripts/export_models.py
```

### "Failed to connect to Qdrant"

Ensure Qdrant is running:
```bash
docker run -p 6333:6333 qdrant/qdrant:latest
```

Or use memory-only mode:
```bash
./target/release/sam-detect-cli image.jpg --skip-qdrant
```

### "libssl.so.3: cannot open shared object"

```bash
# Ubuntu/Debian
sudo apt-get install libssl3

# Or set library path
export LD_LIBRARY_PATH=/usr/lib/x86_64-linux-gnu:$LD_LIBRARY_PATH
```

## Comparison: Python vs Rust

| Aspect | Python | Rust |
|--------|--------|------|
| **Development Speed** | ⭐⭐⭐⭐⭐ | ⭐⭐⭐ |
| **Runtime Performance** | ⭐⭐⭐ | ⭐⭐⭐⭐⭐ |
| **Memory Safety** | ⭐⭐ | ⭐⭐⭐⭐⭐ |
| **Latency Predictability** | ⭐⭐⭐ | ⭐⭐⭐⭐⭐ |
| **Deployment** | Complex | Simple (single binary) |
| **Best For** | Research & Development | Production & Edge Deployment |

## Next Steps

- [ ] Implement advanced mask decoding (currently uses full-image masks)
- [ ] Add GPU memory pooling for reduced allocation overhead
- [ ] Support batch inference for multiple images
- [ ] Add WebSocket server for real-time streaming
- [ ] Optimize ONNX export for different hardware targets

## License

MIT License - same as parent sam-detect project

## Related

- **Python Implementation**: See parent directory
- **Models**: SAM2 (Meta), CLIP (OpenAI), ONNX Runtime (Microsoft)
- **Vector DB**: Qdrant
- **TensorRT**: NVIDIA

## Contributing

Contributions welcome! Areas of interest:

- Performance optimization
- Additional ONNX model support
- Platform-specific tuning (arm64, etc.)
- Web UI / HTTP server
- Streaming/real-time features
