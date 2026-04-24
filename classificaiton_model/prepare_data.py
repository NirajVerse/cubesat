import os
import shutil
import random
import pandas as pd
from config import *

def create_sample_dataset():
    """Create sample dataset structure (for demonstration)"""
    # Create directory structure
    os.makedirs(os.path.join(DATASET_PATH, 'fire'), exist_ok=True)
    os.makedirs(os.path.join(DATASET_PATH, 'non_fire'), exist_ok=True)
    
    print(f"Dataset directory structure created at: {DATASET_PATH}")
    print("Please place your fire images in the 'fire' folder and non-fire images in the 'non_fire' folder")
    print(f"The dataset should contain a total of {TRAIN_SIZE + TEST_SIZE} images")
    print(f"Of which {int(0.54 * (TRAIN_SIZE + TEST_SIZE))} fire images and {int(0.46 * (TRAIN_SIZE + TEST_SIZE))} non-fire images")

if __name__ == "__main__":
    create_sample_dataset()