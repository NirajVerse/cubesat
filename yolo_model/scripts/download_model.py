"""
Download YOLO model weights before training
Run this once to cache the model locally
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'config'))
from yolo_config import *


def download_model():
    """
    Download YOLOv11 model weights
    """
    print("="*60)
    print("DOWNLOADING YOLO MODEL")
    print("="*60)

    try:
        from ultralytics import YOLO

        print(f"\nDownloading {MODEL_NAME} model...")
        print("Size: ~2.5 MB")
        print("This may take 1-2 minutes on first download...\n")

        # Download model
        model = YOLO(f'{MODEL_NAME}.pt')

        print("\n" + "="*60)
        print("SUCCESS: Model downloaded!")
        print("="*60)
        print("\nYou can now run training:")
        print("   python main.py train")
        return True

    except Exception as e:
        print("\n" + "="*60)
        print("ERROR: Could not download model")
        print("="*60)
        print(f"\nError details: {e}")
        print("\nTroubleshooting:")
        print("1. Check internet connection")
        print("2. Try again in a few moments")
        print("3. If persists, try: python -m pip install --upgrade ultralytics")
        return False


if __name__ == '__main__':
    download_model()
