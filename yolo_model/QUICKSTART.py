"""
QUICK START GUIDE
Complete workflow from data preparation to inference
"""

# ============================================================================
#  WHAT YOU NEED FOR YOLO TRAINING (Q&A)
# ============================================================================

## Q: Do I need images with bounding boxes drawn on them?
## A: NO! You need:
##     Images (PNG/JPG) - in YOLODataset/images/
##     Annotations (TXT files) - in YOLODataset/labels/
##     NOT images with boxes already drawn
##
## Your YOLO label files are in format:
##   class_id center_x center_y width height (all normalized 0-1)
##   Example: 0 0.66 0.612 0.613 0.374
##
## The training script will use these annotations to:
##   1. Learn where smoke is in images
##   2. Draw boxes during inference
##   3. Generate visualizations
##

# ============================================================================
#  COMPLETE WORKFLOW
# ============================================================================

STEP 1: INSTALL DEPENDENCIES


cd yolo_model
pip install -r requirements.txt

(Takes ~2 minutes)
Installing: ultralytics, torch, opencv, numpy


STEP 2: PREPARE DATASET (Train/Val/Test Split)


python main.py prepare-dataset

Or with custom split:
python main.py prepare-dataset --train 0.7 --val 0.2

This will:
   Create train/val/test folders
   Copy images and labels to respective folders
   Organize data for training

Results:
  YOLODataset/images/train/     (70% - ~835 images)
  YOLODataset/images/val/       (20% - ~239 images)
  YOLODataset/images/test/      (10% - ~120 images)
  YOLODataset/labels/train/     (matching labels)
  YOLODataset/labels/val/
  YOLODataset/labels/test/


STEP 3: TRAIN YOLO MODEL


python main.py train

  Estimated time: 30-60 minutes (on GPU)
    If using CPU: 2-4 hours

The script will:
   Download YOLOv11 nano model (~2.5 MB)
   Train for 100 epochs
   Save best model to: models/best.pt
   Generate training plots and metrics

Output files:
  models/smoke_detection_YYYYMMDD_HHMMSS/weights/best.pt  (actual weights)
  models/best.pt                                           (copy for easy access)


STEP 4: RUN INFERENCE


On a single image:
  python main.py infer \
    --image path/to/image.jpg \
    --model models/best.pt \
    --save

On multiple images (directory):
  python main.py infer \
    --image path/to/images/ \
    --model models/best.pt \
    --save

Results:
   inference_results/pred_image.jpg (annotated with boxes)
   Console output with detections


# ============================================================================
#  STRUCTURE COMPARISON
# ============================================================================

YOUR CLASSIFICATION PIPELINE vs YOLO PIPELINE

Classification (Already trained):
  Input: Image  Model  Output: "Fire" or "No Fire"
  Use case: Binary classification

YOLO (New pipeline):
  Input: Image  Model  Output: Bounding boxes with confidence
  Use case: Localization + detection


# ============================================================================
#  FILE STRUCTURE
# ============================================================================

yolo_model/
 config/
    yolo_config.py          Settings (model size, thresholds, etc)
 scripts/
    train.py                Train the model
    inference.py            Run predictions
    prepare_dataset.py      Prepare data splits
 models/                     Trained weights (created after training)
    best.pt                 Best model (use for inference)
 inference_results/          Output images with boxes
    pred_image1.jpg         Annotated prediction
 main.py                     Easy entry point
 README.md                   Full documentation
 requirements.txt            Dependencies


# ============================================================================
#  INFERENCE EXAMPLES
# ============================================================================

Example 1: Single image
python main.py infer \
  --image ../Example\ Data\ 1/fire_image.jpg \
  --model models/best.pt \
  --save  print("="*60)


Example 2: Your test images
python main.py infer \
  --image ../YOLODataset/images/test/ \
  --model models/best.pt \
  --save

Example 3: Lower confidence (more detections)
python main.py infer \
  --image ../Example\ Data\ 1/ \
  --model models/best.pt \
  --conf 0.3 \
  --save

Example 4: Higher confidence (fewer false positives)
python main.py infer \
  --image image.jpg \
  --model models/best.pt \
  --conf 0.7


# ============================================================================
#  CONFIGURATION
# ============================================================================

Edit config/yolo_config.py to customize:

MODEL_NAME = 'yolov11n'         # Choose model size
  - 'yolov11n' (nano) - fastest, ~2MB
  - 'yolov11s' (small) - balanced
  - 'yolov11m' (medium) - accurate
  - 'yolov11l' (large) - most accurate

EPOCHS = 100                    # How long to train
BATCH_SIZE = 16                 # Images per iteration
IMGSZ = 512                     # Input image size
INFERENCE_CONF = 0.5            # Confidence threshold


# ============================================================================
#  EXPECTED OUTPUTS
# ============================================================================

After training:
   models/best.pt (~5-7 MB)
   Training plots and metrics

After inference:
   inference_results/pred_*.jpg (with bounding boxes)
   Console output:
      image.jpg
        Smoke regions detected: 2
        [1] Confidence: 85.3%
        [2] Confidence: 72.1%


# ============================================================================
#  FAQ
# ============================================================================

Q: Where do I put my own images to test?
A: Anywhere! Use:
   python main.py infer --image /path/to/image.jpg --model models/best.pt

Q: What if I don't have train/val/test split?
A: Run prepare-dataset first! It creates the split automatically.

Q: How do I know if training is working?
A: Check the output - loss should decrease and accuracy should increase.
   See training plots in models/smoke_detection_*/

Q: Can I improve accuracy?
A: Yes! After initial training:
   - Train more epochs
   - Use larger model (yolov11m instead of yolov11n)
   - Check if annotations are accurate

Q: How long does inference take?
A: ~0.1-0.5 seconds per image (on GPU)

Q: Can I use on CPU?
A: Yes, but slower. Set DEVICE='cpu' in config.
   Inference will be ~1-5 seconds per image.


# ============================================================================
#  NEXT STEPS
# ============================================================================

1. Install dependencies:     pip install -r requirements.txt
2. Prepare dataset:          python main.py prepare-dataset
3. Train model:              python main.py train
4. Run inference:            python main.py infer --image <path> --model models/best.pt
5. Check results:            See inference_results/ folder

All done! 
"""

# This file is documentation. No Python code to execute.
