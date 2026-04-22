# YOLO Configuration
import os

# Paths
DATASET_PATH = '../YOLODataset'
YOLO_MODELS_DIR = './models/'
INFERENCE_RESULTS_DIR = './inference_results/'

# Model Configuration
MODEL_NAME = 'yolov11n'  # yolov11n (nano), yolov11s (small), yolov11m (medium), yolov11l (large)
DEVICE = 'cuda'  # 'cuda' or 'cpu'

# Training Configuration (for future use)
EPOCHS = 100
BATCH_SIZE = 16
IMG_SIZE = 512
IMGSZ = 512
CONFIDENCE_THRESHOLD = 0.5
IOU_THRESHOLD = 0.45

# Classes
CLASSES = ['smoke']
NUM_CLASSES = 1

# Dataset paths (will be updated after train/val/test split)
TRAIN_IMG_DIR = os.path.join(DATASET_PATH, 'images/train')
VAL_IMG_DIR = os.path.join(DATASET_PATH, 'images/val')
TEST_IMG_DIR = os.path.join(DATASET_PATH, 'images/test')

# Inference
INFERENCE_CONF = 0.5  # Confidence threshold for predictions
INFERENCE_IOU = 0.45  # NMS IOU threshold

# Visualization
SAVE_PREDICTIONS = True
DRAW_CONFIDENCE = True
LINE_WIDTH = 2
FONT_SIZE = 0.5

# Color for bounding boxes (BGR format for OpenCV)
BOX_COLOR = (0, 255, 0)  # Green
TEXT_COLOR = (255, 255, 255)  # White
