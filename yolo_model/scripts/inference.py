"""
YOLO Inference Pipeline for Smoke Detection
Predicts bounding boxes on input images and visualizes results
"""

import os
import cv2
import torch
import numpy as np
from pathlib import Path
from ultralytics import YOLO
import sys

# Add config to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'config'))
from yolo_config import *

class SmokeDetector:
    """
    YOLO-based smoke detection and visualization
    """

    def __init__(self, model_path=None, conf_threshold=INFERENCE_CONF):
        """
        Initialize the detector with a trained model

        Args:
            model_path: Path to trained .pt model file
            conf_threshold: Confidence threshold for detections
        """
        self.conf_threshold = conf_threshold
        self.device = DEVICE

        # Load model
        if model_path is None:
            print("⚠️  No model path provided. Please provide a trained model.")
            print(f"   Expected model location: {YOLO_MODELS_DIR}")
            self.model = None
        else:
            try:
                self.model = YOLO(model_path)
                print(f"✅ Model loaded from: {model_path}")
            except Exception as e:
                print(f"❌ Error loading model: {e}")
                self.model = None

    def predict_single_image(self, image_path, visualize=True):
        """
        Run inference on a single image

        Args:
            image_path: Path to input image
            visualize: Whether to draw bounding boxes

        Returns:
            results: Detection results
            annotated_image: Image with bounding boxes (if visualize=True)
        """
        if self.model is None:
            print("❌ Model not loaded. Cannot run inference.")
            return None, None

        if not os.path.exists(image_path):
            print(f"❌ Image not found: {image_path}")
            return None, None

        # Run inference
        results = self.model.predict(
            source=image_path,
            conf=self.conf_threshold,
            device=self.device,
            verbose=False
        )

        # Get annotated image
        if visualize and len(results) > 0:
            result = results[0]
            annotated_image = result.plot()  # YOLO's built-in plotting
        else:
            annotated_image = None

        return results, annotated_image

    def predict_batch(self, image_dir, visualize=True, save_results=False):
        """
        Run inference on multiple images in a directory

        Args:
            image_dir: Directory containing images
            visualize: Whether to draw bounding boxes
            save_results: Whether to save annotated images

        Returns:
            results_dict: Dictionary with results for each image
        """
        if self.model is None:
            print("❌ Model not loaded. Cannot run inference.")
            return {}

        if not os.path.isdir(image_dir):
            print(f"❌ Directory not found: {image_dir}")
            return {}

        # Get all image files
        image_extensions = ('.jpg', '.jpeg', '.png', '.bmp', '.tiff')
        image_files = [f for f in os.listdir(image_dir)
                      if f.lower().endswith(image_extensions)]

        if not image_files:
            print(f"⚠️  No images found in {image_dir}")
            return {}

        print(f"🔍 Found {len(image_files)} images. Running inference...")

        results_dict = {}

        for i, img_file in enumerate(image_files, 1):
            img_path = os.path.join(image_dir, img_file)
            print(f"   [{i}/{len(image_files)}] Processing: {img_file}")

            results, annotated_img = self.predict_single_image(img_path, visualize=visualize)

            if results:
                results_dict[img_file] = {
                    'path': img_path,
                    'results': results[0],
                    'annotated_image': annotated_img,
                    'detections': self._extract_detections(results[0])
                }

                # Save if requested
                if save_results and annotated_img is not None:
                    output_path = os.path.join(
                        INFERENCE_RESULTS_DIR,
                        f'pred_{img_file}'
                    )
                    cv2.imwrite(output_path, annotated_img)
                    print(f"      💾 Saved: {output_path}")

        return results_dict

    def _extract_detections(self, result):
        """
        Extract detection information from YOLO result

        Args:
            result: YOLO result object

        Returns:
            list of detection dictionaries
        """
        detections = []

        if result.boxes is not None:
            for box in result.boxes:
                det = {
                    'class': int(box.cls[0]),
                    'confidence': float(box.conf[0]),
                    'bbox': {
                        'x1': float(box.xyxy[0][0]),
                        'y1': float(box.xyxy[0][1]),
                        'x2': float(box.xyxy[0][2]),
                        'y2': float(box.xyxy[0][3])
                    }
                }
                detections.append(det)

        return detections

    def save_results(self, annotated_image, output_path):
        """
        Save annotated image to file

        Args:
            annotated_image: Numpy array of image with bounding boxes
            output_path: Path to save the image
        """
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        cv2.imwrite(output_path, annotated_image)
        print(f"✅ Saved: {output_path}")

    def print_detections(self, results_dict):
        """
        Print detection summary for all processed images

        Args:
            results_dict: Dictionary with results from predict_batch
        """
        print("\n" + "="*60)
        print("📊 DETECTION SUMMARY")
        print("="*60)

        total_detections = 0
        images_with_smoke = 0

        for img_name, result_info in results_dict.items():
            detections = result_info['detections']
            num_detections = len(detections)
            total_detections += num_detections

            if num_detections > 0:
                images_with_smoke += 1
                print(f"\n📷 {img_name}")
                print(f"   Smoke regions detected: {num_detections}")
                for i, det in enumerate(detections, 1):
                    conf = det['confidence'] * 100
                    print(f"   [{i}] Confidence: {conf:.1f}%")

        print("\n" + "-"*60)
        print(f"Total images processed: {len(results_dict)}")
        print(f"Images with smoke: {images_with_smoke}")
        print(f"Total smoke detections: {total_detections}")
        print("="*60 + "\n")


def main():
    """
    Main inference pipeline - specify image path and run
    """
    import argparse

    parser = argparse.ArgumentParser(
        description='YOLO Smoke Detection Inference'
    )
    parser.add_argument(
        '--model',
        type=str,
        help='Path to trained YOLO model (.pt file)',
        default=None
    )
    parser.add_argument(
        '--image',
        type=str,
        help='Path to input image or directory'
    )
    parser.add_argument(
        '--conf',
        type=float,
        default=INFERENCE_CONF,
        help='Confidence threshold'
    )
    parser.add_argument(
        '--save',
        action='store_true',
        help='Save annotated images'
    )

    args = parser.parse_args()

    # Initialize detector
    detector = SmokeDetector(model_path=args.model, conf_threshold=args.conf)

    if detector.model is None:
        print("\n❌ Cannot proceed without a model. Please provide --model argument.")
        print(f"   Example: python inference.py --model models/best.pt --image path/to/image.jpg")
        return

    # Create results directory
    os.makedirs(INFERENCE_RESULTS_DIR, exist_ok=True)

    # Run inference
    if os.path.isdir(args.image):
        print(f"\n📁 Processing directory: {args.image}")
        results = detector.predict_batch(
            args.image,
            visualize=True,
            save_results=args.save
        )
        detector.print_detections(results)
    else:
        print(f"\n🖼️  Processing single image: {args.image}")
        results, annotated = detector.predict_single_image(args.image, visualize=True)

        if results and annotated is not None:
            detections = detector._extract_detections(results[0])
            print(f"\n✅ Detections: {len(detections)}")
            for i, det in enumerate(detections, 1):
                print(f"   [{i}] Smoke - Confidence: {det['confidence']*100:.1f}%")

            if args.save:
                output_path = os.path.join(
                    INFERENCE_RESULTS_DIR,
                    f'pred_{os.path.basename(args.image)}'
                )
                detector.save_results(annotated, output_path)


if __name__ == '__main__':
    main()
