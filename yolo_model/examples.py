"""
Example: Using the SmokeDetector programmatically
This shows how to integrate the inference pipeline into your own code
"""

import sys
import os

# Add paths
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'scripts'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'config'))

from inference import SmokeDetector
from yolo_config import *


def example_single_image():
    """
    Example 1: Predict on a single image
    """
    print("="*60)
    print("Example 1: Single Image Prediction")
    print("="*60)

    # Initialize detector
    model_path = './models/best.pt'
    detector = SmokeDetector(model_path=model_path, conf_threshold=0.5)

    # Predict
    image_path = '../Example Data 1/your_image.jpg'  # Update with actual path
    results, annotated_image = detector.predict_single_image(
        image_path,
        visualize=True
    )

    if results:
        detections = detector._extract_detections(results[0])
        print(f"\n Found {len(detections)} smoke region(s)")

        for i, det in enumerate(detections, 1):
            print(f"   [{i}] Confidence: {det['confidence']*100:.1f}%")
            print(f"       Box: ({det['bbox']['x1']:.0f}, {det['bbox']['y1']:.0f}) " +
                  f"to ({det['bbox']['x2']:.0f}, {det['bbox']['y2']:.0f})")

        # Save
        if annotated_image is not None:
            output_path = f'{INFERENCE_RESULTS_DIR}/example_single.jpg'
            detector.save_results(annotated_image, output_path)


def example_batch_prediction():
    """
    Example 2: Predict on multiple images in a directory
    """
    print("\n" + "="*60)
    print("Example 2: Batch Prediction")
    print("="*60)

    detector = SmokeDetector(
        model_path='./models/best.pt',
        conf_threshold=0.5
    )

    # Directory with multiple images
    image_dir = '../YOLODataset/images/test/'

    # Run batch prediction
    results = detector.predict_batch(
        image_dir,
        visualize=True,
        save_results=True  # Save annotated images
    )

    # Print summary
    detector.print_detections(results)


def example_custom_inference():
    """
    Example 3: Custom inference with manual post-processing
    """
    print("\n" + "="*60)
    print("Example 3: Custom Processing")
    print("="*60)

    detector = SmokeDetector(model_path='./models/best.pt')
    image_path = '../Example Data 1/fire_image.jpg'  # Update path

    results, annotated = detector.predict_single_image(image_path)

    if results:
        detections = detector._extract_detections(results[0])

        # Custom processing
        high_confidence_detections = [
            d for d in detections if d['confidence'] > 0.7
        ]

        print(f"High confidence detections (>0.7): {len(high_confidence_detections)}")

        for det in high_confidence_detections:
            bbox = det['bbox']
            area = (bbox['x2'] - bbox['x1']) * (bbox['y2'] - bbox['y1'])
            print(f"  Confidence: {det['confidence']*100:.1f}%, Area: {area:.0f} px")


def example_with_different_thresholds():
    """
    Example 4: Try different confidence thresholds
    """
    print("\n" + "="*60)
    print("Example 4: Threshold Comparison")
    print("="*60)

    image_path = '../Example Data 1/fire_image.jpg'  # Update path
    thresholds = [0.3, 0.5, 0.7, 0.9]

    for conf in thresholds:
        detector = SmokeDetector(model_path='./models/best.pt', conf_threshold=conf)
        results, _ = detector.predict_single_image(image_path, visualize=False)

        if results:
            detections = detector._extract_detections(results[0])
            print(f"   Confidence threshold {conf}: {len(detections)} detections")


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Run inference examples')
    parser.add_argument(
        '--example',
        type=int,
        choices=[1, 2, 3, 4],
        default=1,
        help='Which example to run'
    )

    args = parser.parse_args()

    if args.example == 1:
        example_single_image()
    elif args.example == 2:
        example_batch_prediction()
    elif args.example == 3:
        example_custom_inference()
    elif args.example == 4:
        example_with_different_thresholds()
