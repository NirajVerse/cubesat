# YOLO Smoke Detection Pipeline

This folder contains the complete YOLO object detection pipeline for smoke/fire detection from satellite imagery.

##  Folder Structure

```
yolo_model/
 config/
    yolo_config.py         # Configuration settings
 scripts/
    inference.py            # Inference on images
    train.py                # Train YOLO model
    prepare_dataset.py      # Prepare dataset splits
 models/                     # Trained model weights (generated after training)
 inference_results/          # Prediction outputs (generated after inference)
 requirements.txt            # Python dependencies
```

##  Quick Start

### 1. Install Dependencies

```bash
cd yolo_model
pip install -r requirements.txt
```

### 2. Prepare Dataset (One-time setup)

Before training, split your data into train/val/test directories:

```bash
python scripts/prepare_dataset.py \
  --images ../YOLODataset/images \
  --labels ../YOLODataset/labels \
  --train 0.7 \
  --val 0.2
```

This creates:
- `YOLODataset/images/train/` - 70% for training
- `YOLODataset/images/val/` - 20% for validation
- `YOLODataset/images/test/` - 10% for testing

### 3. Train Model

```bash
python scripts/train.py
```

Or with custom dataset:
```bash
python scripts/train.py --dataset-yaml path/to/dataset.yaml
```

Training will:
- Download YOLOv11 pretrained weights (nano version - lightweight)
- Train for 100 epochs
- Save best model to `models/best.pt`
- Generate training plots and metrics

 **Training time**: ~30-60 minutes on GPU (varies by hardware)

### 4. Run Inference

#### Single Image:
```bash
python scripts/inference.py \
  --model models/best.pt \
  --image path/to/image.jpg \
  --save
```

#### Batch Processing (Directory):
```bash
python scripts/inference.py \
  --model models/best.pt \
  --image path/to/images/ \
  --save
```

#### With Custom Confidence:
```bash
python scripts/inference.py \
  --model models/best.pt \
  --image path/to/image.jpg \
  --conf 0.5 \
  --save
```

##  Output

### Inference Results

- **Annotated images**: Saved to `inference_results/` with `pred_` prefix
- **Bounding boxes**: Green rectangles around detected smoke
- **Confidence scores**: Displayed on boxes

### Console Output

```
 image_name.jpg
   Smoke regions detected: 2
   [1] Confidence: 85.3%
   [2] Confidence: 72.1%
```

##  Configuration

Edit `config/yolo_config.py` to customize:

```python
MODEL_NAME = 'yolov11n'           # Model size (nano/small/medium/large)
IMGSZ = 512                        # Image size for inference
INFERENCE_CONF = 0.5               # Confidence threshold
DEVICE = 'cuda'                    # 'cuda' or 'cpu'
```

##  Model Sizes

| Model | Size | Speed | Accuracy |
|-------|------|-------|----------|
| yolov11n | ~2MB | Fast | Good |
| yolov11s | ~7MB | Medium | Better |
| yolov11m | ~20MB | Slower | Best |
| yolov11l | ~25MB | Slowest | Excellent |

**Start with `yolov11n` (nano) - fastest training and inference**

##  Understanding Results

### Bounding Box Format
- **Green box**: Detected smoke region
- **Text**: Class name + Confidence percentage
- **Coordinates**: Pixel location of detection

### Confidence Score
- 0.0 - 0.3: Low confidence (likely false positive)
- 0.3 - 0.6: Medium confidence
- 0.6 - 1.0: High confidence (likely true positive)

Adjust `--conf` parameter to filter detections by confidence.

##  Troubleshooting

### No detections?
- Lower confidence threshold: `--conf 0.3`
- Check if model is trained (should have `models/best.pt`)
- Verify image quality/format

### Out of memory?
- Reduce `BATCH_SIZE` in config
- Use smaller model (`yolov11n`)
- Reduce `IMGSZ`

### Model not training?
- Check dataset structure: `images/train/`, `labels/train/`, `images/val/`, `labels/val/`
- Verify all images have matching label files
- Check dataset paths in `dataset.yaml`

##  Training Data Format

Your dataset uses **YOLO format**:

```
image1.jpg   image1.txt
image2.jpg   image2.txt
```

Label file (image1.txt):
```
0 0.66 0.612 0.613 0.374
```

Where:
- `0` = class ID (smoke = 0)
- `0.66` = center X (normalized 0-1)
- `0.612` = center Y (normalized 0-1)
- `0.613` = width (normalized 0-1)
- `0.374` = height (normalized 0-1)

##  Next Steps

1. **Prepare dataset**  `prepare_dataset.py`
2. **Train model**  `train.py` (30-60 mins)
3. **Evaluate results**  Check training plots
4. **Run inference**  `inference.py` on new images
5. **Fine-tune**  Adjust confidence threshold based on results

##  Tips

- Start with small model (`yolov11n`) for faster experimentation
- Use high confidence threshold (0.6+) for production
- Visualize predictions before deployment
- Keep training logs for debugging

##  Questions?

Check the main README in the parent directory or see `train val scripts/README` for classification pipeline comparison.
