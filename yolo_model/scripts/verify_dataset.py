"""
Verify that images and labels are properly paired
"""

import os
import sys
from pathlib import Path

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'config'))
from yolo_config import *


def verify_image_label_pairs(images_dir, labels_dir):
    """
    Check if every image has a matching label file
    """
    print("="*60)
    print("VERIFYING IMAGE-LABEL PAIRS")
    print("="*60)

    # Get all image files
    image_extensions = ('.jpg', '.jpeg', '.png', '.bmp', '.tiff')
    image_files = set([f for f in os.listdir(images_dir)
                      if f.lower().endswith(image_extensions)])

    # Get all label files
    label_files = set([f for f in os.listdir(labels_dir)
                      if f.lower().endswith('.txt')])

    print(f"\nTotal images: {len(image_files)}")
    print(f"Total labels: {len(label_files)}")

    # Check matching
    missing_labels = []
    missing_images = []
    paired_count = 0

    # Check each image has label
    for img_file in image_files:
        label_file = os.path.splitext(img_file)[0] + '.txt'
        if label_file in label_files:
            paired_count += 1
        else:
            missing_labels.append(img_file)

    # Check each label has image
    for lbl_file in label_files:
        img_base = os.path.splitext(lbl_file)[0]
        found = False
        for img_file in image_files:
            if os.path.splitext(img_file)[0] == img_base:
                found = True
                break
        if not found:
            missing_images.append(lbl_file)

    # Report
    print(f"\nPaired: {paired_count}")
    print(f"Missing labels: {len(missing_labels)}")
    print(f"Missing images: {len(missing_images)}")

    if missing_labels:
        print("\nImages WITHOUT matching labels:")
        for i, f in enumerate(missing_labels[:10], 1):
            print(f"   {i}. {f}")
        if len(missing_labels) > 10:
            print(f"   ... and {len(missing_labels)-10} more")

    if missing_images:
        print("\nLabels WITHOUT matching images:")
        for i, f in enumerate(missing_images[:10], 1):
            print(f"   {i}. {f}")
        if len(missing_images) > 10:
            print(f"   ... and {len(missing_images)-10} more")

    # Summary
    print("\n" + "="*60)
    if paired_count == len(image_files) and len(missing_images) == 0:
        print("SUCCESS: All images have matching labels!")
        print("Dataset is ready for training.")
        return True
    else:
        print("ERROR: Dataset has mismatches!")
        print("Please fix before training.")
        return False


def verify_label_format(labels_dir, num_samples=5):
    """
    Check label files have correct YOLO format
    """
    print("\n" + "="*60)
    print("VERIFYING LABEL FORMAT")
    print("="*60)

    label_files = [f for f in os.listdir(labels_dir) if f.endswith('.txt')]

    if not label_files:
        print("No label files found!")
        return False

    print(f"\nChecking format of {num_samples} sample labels...\n")

    all_valid = True
    for i, label_file in enumerate(label_files[:num_samples], 1):
        label_path = os.path.join(labels_dir, label_file)

        print(f"[{i}] {label_file}")

        with open(label_path, 'r') as f:
            lines = f.readlines()

        if not lines:
            print("   WARNING: Empty label file!")
            all_valid = False
            continue

        for j, line in enumerate(lines, 1):
            parts = line.strip().split()

            if len(parts) != 5:
                print(f"   ERROR: Line {j} has {len(parts)} values (need 5)")
                print(f"   Content: {line.strip()}")
                all_valid = False
                continue

            try:
                class_id = int(parts[0])
                center_x = float(parts[1])
                center_y = float(parts[2])
                width = float(parts[3])
                height = float(parts[4])

                # Validate ranges
                if not (0 <= center_x <= 1):
                    print(f"   ERROR: center_x out of range: {center_x}")
                    all_valid = False
                if not (0 <= center_y <= 1):
                    print(f"   ERROR: center_y out of range: {center_y}")
                    all_valid = False
                if not (0 <= width <= 1):
                    print(f"   ERROR: width out of range: {width}")
                    all_valid = False
                if not (0 <= height <= 1):
                    print(f"   ERROR: height out of range: {height}")
                    all_valid = False

                print(f"   Line {j}: VALID")
                print(f"      Class: {class_id}, Box: ({center_x:.3f}, {center_y:.3f}, {width:.3f}, {height:.3f})")

            except ValueError as e:
                print(f"   ERROR: Could not parse line {j}: {e}")
                print(f"   Content: {line.strip()}")
                all_valid = False

    print("\n" + "="*60)
    if all_valid:
        print("SUCCESS: Label format is correct!")
        return True
    else:
        print("ERROR: Some labels have format issues!")
        return False


def main():
    import argparse

    parser = argparse.ArgumentParser(description='Verify dataset integrity')
    parser.add_argument('--images', type=str, default=os.path.join(DATASET_PATH, 'images'),
                       help='Images directory')
    parser.add_argument('--labels', type=str, default=os.path.join(DATASET_PATH, 'labels'),
                       help='Labels directory')
    parser.add_argument('--samples', type=int, default=5,
                       help='Number of label samples to verify')

    args = parser.parse_args()

    # Verify pairs
    pairs_valid = verify_image_label_pairs(args.images, args.labels)

    # Verify format
    format_valid = verify_label_format(args.labels, args.samples)

    print("\n" + "="*60)
    if pairs_valid and format_valid:
        print("READY FOR TRAINING!")
    else:
        print("FIX ISSUES BEFORE TRAINING")
    print("="*60)


if __name__ == '__main__':
    main()
