# YOLO Configuration
import os

# Paths
DATASET_PATH = '../YOLODataset'
YOLO_MODELS_DIR = './models/'
INFERENCE_RESULTS_DIR = './inference_results/'

# Model Configuration
MODEL_NAME = 'yolo26n'  # yolov11n (nano), yolov11s (small), yolov11m (medium), yolov11l (large)
DEVICE = 'cuda'  # 'cuda' or 'cpu'

# Training Configuration (for future use)
EPOCHS = 200
BATCH_SIZE = 32
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
<<<<<<< HEAD
INFERENCE_CONF = 0.3  # Confidence threshold for predictions
INFERENCE_IOU = 0.1  # NMS IOU threshold
=======
INFERENCE_CONF = 0.25  # Confidence threshold for predictions
INFERENCE_IOU = 0.2  # NMS IOU threshold
>>>>>>> 2c4f0b8 (adeded IOU_Infer)

# Visualization
SAVE_PREDICTIONS = True
DRAW_CONFIDENCE = True
LINE_WIDTH = 2
FONT_SIZE = 0.5

# Color for bounding boxes (BGR format for OpenCV)
BOX_COLOR = (0, 255, 0)  # Green
TEXT_COLOR = (255, 255, 255)  # White
