"""
Dataset preparation utility for YOLO training
Splits dataset into train/val/test directories
"""

import os
import shutil
from pathlib import Path
import random
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'config'))
from yolo_config import *


def split_dataset(images_dir, labels_dir, train_ratio=0.7, val_ratio=0.2, test_ratio=0.1):
    """
    Split dataset into train/val/test directories

    Args:
        images_dir: Directory containing all images
        labels_dir: Directory containing all labels
        train_ratio: Proportion for training (default 0.7)
        val_ratio: Proportion for validation (default 0.2)
        test_ratio: Proportion for testing (default 0.1)
    """
    print("="*60)
    print(" DATASET SPLITTING PIPELINE")
    print("="*60)

    # Verify ratios sum to 1
    if not abs((train_ratio + val_ratio + test_ratio) - 1.0) < 0.001:
        print(" Ratios must sum to 1.0")
        return

    print(f"\n Split Ratios:")
    print(f"   Train: {train_ratio*100:.0f}%")
    print(f"   Val: {val_ratio*100:.0f}%")
    print(f"   Test: {test_ratio*100:.0f}%")

    # Get all image files
    image_extensions = ('.jpg', '.jpeg', '.png', '.bmp', '.tiff')
    image_files = [f for f in os.listdir(images_dir)
                  if f.lower().endswith(image_extensions)]

    if not image_files:
        print(f"\n No images found in {images_dir}")
        return

    print(f"\n Found {len(image_files)} images")

    # Shuffle and split
    random.shuffle(image_files)
    total = len(image_files)
    train_count = int(total * train_ratio)
    val_count = int(total * val_ratio)

    train_files = image_files[:train_count]
    val_files = image_files[train_count:train_count + val_count]
    test_files = image_files[train_count + val_count:]

    print(f"   Train: {len(train_files)} images")
    print(f"   Val: {len(val_files)} images")
    print(f"   Test: {len(test_files)} images")

    # Create directories
    print("\n Creating directories...")
    splits = {
        'train': train_files,
        'val': val_files,
        'test': test_files
    }

    for split, files in splits.items():
        # Create image directory
        img_split_dir = os.path.join(images_dir, split)
        os.makedirs(img_split_dir, exist_ok=True)

        # Create labels directory
        lbl_split_dir = os.path.join(labels_dir, split)
        os.makedirs(lbl_split_dir, exist_ok=True)

        print(f"    Created {split} directories")

        # Copy files
        print(f"    Copying {split} files...")
        for i, img_file in enumerate(files, 1):
            # Get corresponding label file
            label_file = os.path.splitext(img_file)[0] + '.txt'

            # Copy image
            src_img = os.path.join(images_dir, img_file)
            dst_img = os.path.join(img_split_dir, img_file)
            shutil.copy(src_img, dst_img)

            # Copy label if exists
            src_lbl = os.path.join(labels_dir, label_file)
            if os.path.exists(src_lbl):
                dst_lbl = os.path.join(lbl_split_dir, label_file)
                shutil.copy(src_lbl, dst_lbl)

            if i % 100 == 0:
                print(f"      [{i}/{len(files)}] copied")

        print(f"    Copied all {split} files")

    print("\n" + "="*60)
    print(" Dataset split completed!")
    print("="*60)
    print("\nDataset is ready for training!")


def main():
    """
    Main dataset preparation
    """
    import argparse

    parser = argparse.ArgumentParser(description='Prepare dataset for YOLO training')
    parser.add_argument(
        '--images',
        type=str,
        default=os.path.join(DATASET_PATH, 'images'),
        help='Path to images directory'
    )
    parser.add_argument(
        '--labels',
        type=str,
        default=os.path.join(DATASET_PATH, 'labels'),
        help='Path to labels directory'
    )
    parser.add_argument(
        '--train',
        type=float,
        default=0.7,
        help='Training ratio'
    )
    parser.add_argument(
        '--val',
        type=float,
        default=0.2,
        help='Validation ratio'
    )

    args = parser.parse_args()

    test_ratio = 1.0 - args.train - args.val

    split_dataset(
        images_dir=args.images,
        labels_dir=args.labels,
        train_ratio=args.train,
        val_ratio=args.val,
        test_ratio=test_ratio
    )


if __name__ == '__main__':
    main()
