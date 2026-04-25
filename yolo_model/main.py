#!/usr/bin/env python
"""
YOLO Smoke Detection - Main Entry Point
Simple interface to train and run inference
"""

import os
import sys
import argparse
from pathlib import Path

# Add scripts to path
script_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(script_dir, 'scripts'))
sys.path.insert(0, os.path.join(script_dir, 'config'))

from yolo_config import *


def main():
    parser = argparse.ArgumentParser(
        description='YOLO Smoke Detection Pipeline',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Train model
  python main.py train

  # Inference on single image
  python main.py infer --image path/to/image.jpg --model models/best.pt

  # Inference on directory
  python main.py infer --image path/to/images/ --model models/best.pt --save

  # Prepare dataset
  python main.py prepare-dataset
        """
    )

    subparsers = parser.add_subparsers(dest='command', help='Command to run')

    # Train command
    train_parser = subparsers.add_parser('train', help='Train YOLO model')
    train_parser.add_argument(
        '--dataset-yaml',
        type=str,
        default=None,
        help='Path to dataset.yaml'
    )

    # Inference command
    infer_parser = subparsers.add_parser('infer', help='Run inference on images')
    infer_parser.add_argument( 
        '--image',
        type=str,
        required=True,
        help='Path to image or directory'
    )
    infer_parser.add_argument(
        '--model',
        type=str,
        required=True,
        help='Path to trained model (.pt file)'
    )
    infer_parser.add_argument(
        '--conf',
        type=float,
        default=INFERENCE_CONF,
        help='Confidence threshold'
    )
    infer_parser.add_argument(
        '--save',
        action='store_true',
        help='Save annotated images'
    )

    # Prepare dataset command
    prep_parser = subparsers.add_parser('prepare-dataset', help='Prepare dataset for training')
    prep_parser.add_argument(
        '--images',
        type=str,
        default=os.path.join(DATASET_PATH, 'images'),
        help='Path to images directory'
    )
    prep_parser.add_argument(
        '--labels',
        type=str,
        default=os.path.join(DATASET_PATH, 'labels'),
        help='Path to labels directory'
    )
    prep_parser.add_argument(
        '--train',
        type=float,
        default=0.7,
        help='Training ratio'
    )
    prep_parser.add_argument(
        '--val',
        type=float,
        default=0.2,
        help='Validation ratio'
    )

    args = parser.parse_args()

    if args.command is None:
        parser.print_help()
        return

    if args.command == 'train':
        from train import train_yolo
        train_yolo(dataset_yaml_path=args.dataset_yaml)

    elif args.command == 'infer':
        from inference import SmokeDetector

        detector = SmokeDetector(model_path=args.model, conf_threshold=args.conf)

        if detector.model is None:
            print(" Failed to load model")
            return

        os.makedirs(INFERENCE_RESULTS_DIR, exist_ok=True)

        if os.path.isdir(args.image):
            print(f"\n Processing directory: {args.image}")
            results = detector.predict_batch(
                args.image,
                visualize=True,
                save_results=args.save
            )
            detector.print_detections(results)
        else:
            print(f"\n  Processing image: {args.image}")
            results, annotated = detector.predict_single_image(args.image, visualize=True)

            if results and annotated is not None:
                detections = detector._extract_detections(results[0])
                print(f"\n Detections: {len(detections)}")
                for i, det in enumerate(detections, 1):
                    conf = det['confidence'] * 100
                    print(f"   [{i}] Smoke - Confidence: {conf:.1f}%")

                if args.save:
                    output_path = os.path.join(
                        INFERENCE_RESULTS_DIR,
                        f'pred_{os.path.basename(args.image)}'
                    )
                    detector.save_results(annotated, output_path)

    elif args.command == 'prepare-dataset':
        from prepare_dataset import split_dataset

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
