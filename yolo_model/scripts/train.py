"""
YOLO Training Script
Train YOLO(different version) model for smoke detection
"""

import os
import torch
import sys
from pathlib import Path
from ultralytics import YOLO
from datetime import datetime

# Add config to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'config'))
from yolo_config import *


def download_model(model_name):
    """
    Ensure YOLO model is downloaded before training
    """
    print(f"\n Preparing {model_name} model...")
    print("   (If first time, will download ~2.5MB - please wait...)")

    try:
        from ultralytics import YOLO
        model = YOLO(f'{model_name}.pt')
        print(f"   Downloaded/loaded successfully!")
        return model
    except Exception as e:
        print(f"   ERROR: Could not load model: {e}")
        print(f"   Check internet connection and try again")
        return None


def check_dataset_structure():
    """
    Check if dataset has proper train/val/test split
    """
    print(" Checking dataset structure...")

    required_dirs = [
        os.path.join(DATASET_PATH, 'images/train'),
        os.path.join(DATASET_PATH, 'images/val'),
        os.path.join(DATASET_PATH, 'labels/train'),
        os.path.join(DATASET_PATH, 'labels/val'),
    ]

    all_exist = True
    for dir_path in required_dirs:
        if os.path.isdir(dir_path):
            num_files = len(os.listdir(dir_path))
            print(f"    {dir_path}: {num_files} files")
        else:
            print(f"    {dir_path}: NOT FOUND")
            all_exist = False

    return all_exist


def create_dataset_yaml():
    """
    Create dataset.yaml for YOLO training
    """
    print("\n Creating dataset.yaml...")

    dataset_yaml = f"""path: {os.path.abspath(DATASET_PATH)}
train: {os.path.abspath(os.path.join(DATASET_PATH, 'images/train'))}
val: {os.path.abspath(os.path.join(DATASET_PATH, 'images/val'))}
test: {os.path.abspath(os.path.join(DATASET_PATH, 'images/test'))}

nc: {NUM_CLASSES}
names: {CLASSES}
"""

    yaml_path = os.path.join(DATASET_PATH, 'dataset_for_training.yaml')
    with open(yaml_path, 'w') as f:
        f.write(dataset_yaml)

    print(f"    Created: {yaml_path}")
    return yaml_path


def train_yolo(dataset_yaml_path=None):
    """
    Train YOLOv11 model for smoke detection

    Args:
        dataset_yaml_path: Path to dataset.yaml file
    """
    print("="*60)
    print(" YOLO TRAINING PIPELINE")
    print("="*60)

    # Check dataset
    if not check_dataset_structure():
        print("\n  Dataset structure incomplete!")
        print("   Please split your dataset into train/val folders first.")
        print("   See Split_dataset.ipynb for help.")
        return

    # Create dataset yaml if not provided
    if dataset_yaml_path is None:
        dataset_yaml_path = create_dataset_yaml()

    # Create models directory
    os.makedirs(YOLO_MODELS_DIR, exist_ok=True)

    # Initialize YOLO model
    print(f"\n Loading {MODEL_NAME} model...")
    print("   (First time will download ~2.5MB - this may take 1-2 minutes)")
    model = download_model(MODEL_NAME)

    if model is None:
        print("\n Unable to download/load model. Check internet connection.")
        return

    # Create run name with timestamp
    run_name = f"smoke_detection_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

    # Training parameters
    print(f"\n  Training Configuration:")
    print(f"   Model: {MODEL_NAME}")
    print(f"   Image Size: {IMGSZ}")
    print(f"   Batch Size: {BATCH_SIZE}")
    print(f"   Epochs: {EPOCHS}")
    print(f"   Device: {DEVICE}")
    print(f"   Confidence Threshold: {INFERENCE_CONF}")

    # Start training
    print(f"\n Starting training...")
    print(f"   Run name: {run_name}")

    results = model.train(
        data=dataset_yaml_path,
        epochs=EPOCHS,
        imgsz=IMGSZ,
        batch=BATCH_SIZE,
        device=DEVICE,
        patience=15,
        save=True,
        project=YOLO_MODELS_DIR,
        name=run_name,
        pretrained=True,
        verbose=True,
        conf=INFERENCE_CONF,
        iou=IOU_THRESHOLD,
        augment = True
    )

    # Save best model with custom name
    print(f"\n Saving best model...")
    best_model_path = os.path.join(YOLO_MODELS_DIR, 'best.pt')
    run_best_path = os.path.join(YOLO_MODELS_DIR, run_name, 'weights', 'best.pt')

    if os.path.exists(run_best_path):
        os.system(f'cp {run_best_path} {best_model_path}')
        print(f"    Saved: {best_model_path}")

    print("\n" + "="*60)
    print(" Training completed!")
    print("="*60)
    print(f"\nTo run inference, use:")
    print(f"   python inference.py --model {best_model_path} --image <path_to_image>")


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Train YOLO model for smoke detection')
    parser.add_argument(
        '--dataset-yaml',
        type=str,
        default=None,
        help='Path to dataset.yaml file'
    )

    args = parser.parse_args()

    train_yolo(dataset_yaml_path=args.dataset_yaml)
