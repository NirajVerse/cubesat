import os

# Paths
DATASET_PATH = '../YOLODataset'
YOLO_MODELS_DIR = './models/'
INFERENCE_RESULTS_DIR = './inference_folder/'

# Model Configuration
MODEL_NAME = 'yolo26n'
DEVICE = 'cuda'

# Training Configuration
EPOCHS = 100
BATCH_SIZE = 64
IMG_SIZE = 512
IMGSZ = 512
CONFIDENCE_THRESHOLD = 0.5
IOU_THRESHOLD = 0.45

# MLOps Configuration
USE_MLFLOW = True
MLFLOW_TRACKING_URI = "http://localhost:5000"  # or 'file:./mlruns'
MLFLOW_EXPERIMENT_NAME = "YOLO_Smoke_Detection"

# Training augmentation
AUGMENT = True
MOSAIC = 1.0
MIXUP = 0.0
COPY_PASTE = 0.0
FLIP_LR = 0.5
FLIP_UD = 0.0
DEGREES = 0.0
TRANSLATE = 0.1
SCALE = 0.5
HSV_H = 0.015
HSV_S = 0.7
HSV_V = 0.4

# Classes
CLASSES = ['smoke']
NUM_CLASSES = 1

# Dataset paths
TRAIN_IMG_DIR = os.path.join(DATASET_PATH, 'images/train')
VAL_IMG_DIR = os.path.join(DATASET_PATH, 'images/val')
TEST_IMG_DIR = os.path.join(DATASET_PATH, 'images/test')

# Inference
INFERENCE_CONF = 0.2
INFERENCE_IOU = 0.2

# Visualization
SAVE_PREDICTIONS = True
DRAW_CONFIDENCE = True
LINE_WIDTH = 2
FONT_SIZE = 0.5

# Color for bounding boxes (BGR format for OpenCV)
BOX_COLOR = (255, 0, 0)
TEXT_COLOR = (255, 255, 255)