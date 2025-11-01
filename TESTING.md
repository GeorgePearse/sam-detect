# Testing sam-detect

This document explains how to test and evaluate the sam-detect system.

## Setup

### 1. Download COCO Dataset

The COCO val2017 dataset (5K images) is already downloaded at:
```
./datasets/coco-minitrain/
```

If you need to re-download it:
```bash
python download_coco_minitrain.py
```

This downloads:
- `val2017/` - 5,000 validation images
- `annotations/instances_val2017.json` - Object instance annotations

### 2. Install Dependencies

Ensure you have the required packages:
```bash
pip install pillow tqdm
```

## Running Tests

### Quick Test: Basic Pipeline

Run a quick test on 10 images with 3 labeled examples:
```bash
python test_pipeline.py
```

This will:
1. **Phase 1**: Add labeled examples from the dataset
   - Extracts 3 images from COCO
   - Adds all unique object categories as labeled examples
   - Stores embeddings in memory

2. **Phase 2**: Run detection on 10 test images
   - Segments each image
   - Embeds segmented regions
   - Searches for nearest neighbors in label database
   - Reports detected objects and confidence scores

3. **Results**: Saves detailed results to `test_results.json`

### Custom Test: Adjustable Parameters

To run with different parameters:
```bash
python test_pipeline.py --num-images 20 --num-examples 5
```

Or modify the script directly:
```python
results = run_test_pipeline(
    dataset_dir="./datasets/coco-minitrain",
    num_images=50,      # Process 50 images
    num_examples=10,    # Add 10 labeled examples
)
```

## Test Output

The test pipeline produces:

### Console Output
Shows real-time progress:
```
Phase 1: Adding 3 labeled examples...
  ✓ Added example: carrot (image 397133)
  ✓ Added example: oven (image 397133)
  ...

Phase 2: Running detection on 10 images...
  [1/10] Image 397133: 1 detections | GT: bottle, dining table, person
  ...
```

### Results File: `test_results.json`
Structured output with:
- Dataset statistics
- Examples added
- Detection metrics per image
- Sample predictions with confidence scores and matches

Example structure:
```json
{
  "dataset_info": {
    "num_images": 5000,
    "num_annotations": 36781,
    "num_categories": 80
  },
  "examples_added": 24,
  "detection_stats": {
    "total_images": 10,
    "total_detections": 10,
    "avg_detections_per_image": 1.0,
    "detection_rate": 100.0
  },
  "all_detections": [...]
}
```

## API Usage

### COCODataset

Load and iterate over COCO dataset:
```python
from sam_detect.coco_utils import COCODataset

dataset = COCODataset(
    images_dir="./datasets/coco-minitrain/val2017",
    annotations_file="./datasets/coco-minitrain/annotations/instances_val2017.json"
)

# Get dataset info
info = dataset.get_dataset_info()
print(f"Images: {info['num_images']}, Categories: {info['num_categories']}")

# Iterate over images with annotations
for image_id, image, annotations in dataset:
    labels = dataset.get_annotation_labels(annotations)
    print(f"Image {image_id}: {labels}")
```

### Test Pipeline

Run detection with custom configuration:
```python
from test_pipeline import run_test_pipeline, print_results

results = run_test_pipeline(
    dataset_dir="./datasets/coco-minitrain",
    num_images=20,
    num_examples=5,
)

print_results(results)
```

### Visualization

Visualize detection results (in visualization module):
```python
from sam_detect.visualization import visualize_detection_result

# Visualize detections on an image
vis_image = visualize_detection_result(image, detections)
vis_image.save("detection_result.png")
```

## Understanding Results

### Detection Metrics

- **Total detections**: Number of object instances found
- **Avg detections/image**: Average objects per image
- **Detection rate**: Percentage of images with ≥1 detection
- **Ground truth objects**: Total annotated objects in test set

### Confidence & Matching

Each detection includes:
- `label`: Predicted object class
- `confidence`: Match confidence (0.0-1.0)
- `matches`: Top-K nearest neighbors from labeled examples
  - `label`: Category name
  - `score`: Similarity score

### Interpreting Output

```
Image 397133:
  Detections: 1
  Ground truth: bottle, dining table, person, knife, ...
    - carrot (1.00) (matched: carrot @ 1.00)
```

This means:
- 1 object was detected (carrot)
- 19 objects were actually in the image
- High confidence match (1.00 score) to our labeled carrot example

## Troubleshooting

### Dataset not found
```
Error: Dataset not found at ./datasets/coco-minitrain
```
→ Run `python download_coco_minitrain.py` to download the dataset

### Low detection rates
The `NaiveSegmenter` used in the test provides basic segmentation. For better results:
- Consider integrating SAM2 or a better segmentation backend
- Add more labeled examples for few-shot learning
- Try different embedding models (CLIP vs average color)

### Memory issues with large datasets
Reduce `num_images` or process in batches:
```python
# Process in smaller batches
for batch_start in range(0, 5000, 100):
    results = run_test_pipeline(num_images=100, ...)
```

## Performance Considerations

| Component | Time/Image |
|-----------|-----------|
| Segmentation (NaiveSegmenter) | ~10ms |
| Embedding (AverageColorEmbedder) | ~1ms |
| Vector search (k=3) | <1ms |
| **Total** | **~11ms** |

For SAM2 or CLIP embeddings, expect higher latencies (100s-1000s ms).

## Next Steps

1. **Integrate SAM2**: Replace `NaiveSegmenter` with proper instance segmentation
2. **Add CLIP embeddings**: Use `CLIPEmbedder` for better semantic matching
3. **Use QDrant**: Store examples in persistent vector database
4. **Evaluate on full dataset**: Test on all 5K images with different configurations
5. **Benchmark metrics**: Compute mAP, F1, or custom metrics against ground truth
