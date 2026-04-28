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

    def __init__(self, model_path=None, conf_threshold=INFERENCE_CONF, iou_threshold=INFERENCE_IOU):
        """
        Initialize the detector with a trained model

        Args:
            model_path: Path to trained .pt model file
            conf_threshold: Confidence threshold for detections
            iou_threshold: IoU threshold for NMS filtering
        """
        self.conf_threshold = conf_threshold
        self.iou_threshold = iou_threshold
        self.device = DEVICE
        self.last_detections = []
        print(f" Inference thresholds -> conf: {self.conf_threshold}, iou: {self.iou_threshold}")

        # Load model
        if model_path is None:
            print("  No model path provided. Please provide a trained model.")
            print(f"   Expected model location: {YOLO_MODELS_DIR}")
            self.model = None
        else:
            try:
                self.model = YOLO(model_path)
                print(f" Model loaded from: {model_path}")
            except Exception as e:
                print(f" Error loading model: {e}")
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
            print(" Model not loaded. Cannot run inference.")
            return None, None

        if not os.path.exists(image_path):
            print(f" Image not found: {image_path}")
            return None, None

        # Run inference
        results = self.model.predict(
            source=image_path,
            conf=self.conf_threshold,
            iou=self.iou_threshold,
            device=self.device,
            verbose=False
        )

        # Extract raw detections and merge near-touching smoke boxes.
        detections = self._extract_detections(results[0]) if len(results) > 0 else []
        detections = self._merge_adjacent_detections(detections)
        self.last_detections = detections

        # Get annotated image
        if visualize and len(results) > 0:
            image = cv2.imread(image_path)
            annotated_image = self._draw_detections(image, detections) if image is not None else None
        else:
            annotated_image = None

        return results, annotated_image

    def _draw_detections(self, image, detections):
        """
        Draw merged detections on image.

        Args:
            image: BGR image array
            detections: list of detection dictionaries

        Returns:
            Annotated image
        """
        if image is None:
            return None

        line_width = int(LINE_WIDTH) if LINE_WIDTH else 2

        for det in detections:
            x1 = int(round(det['bbox']['x1']))
            y1 = int(round(det['bbox']['y1']))
            x2 = int(round(det['bbox']['x2']))
            y2 = int(round(det['bbox']['y2']))
            conf = det['confidence']

            cv2.rectangle(image, (x1, y1), (x2, y2), BOX_COLOR, line_width)
            label = f"smoke {conf:.2f}"
            (text_w, text_h), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, FONT_SIZE, 2)
            text_y1 = max(0, y1 - text_h - baseline - 4)
            text_y2 = max(text_h + baseline + 4, y1)
            cv2.rectangle(image, (x1, text_y1), (x1 + text_w + 6, text_y2), BOX_COLOR, -1)
            cv2.putText(
                image,
                label,
                (x1 + 3, text_y2 - baseline - 2),
                cv2.FONT_HERSHEY_SIMPLEX,
                FONT_SIZE,
                TEXT_COLOR,
                1,
                cv2.LINE_AA,
            )

        return image

    def _boxes_should_merge(self, box_a, box_b):
        """
        Decide whether two smoke boxes represent one plume split into fragments.
        """
        ax1, ay1, ax2, ay2 = box_a['x1'], box_a['y1'], box_a['x2'], box_a['y2']
        bx1, by1, bx2, by2 = box_b['x1'], box_b['y1'], box_b['x2'], box_b['y2']

        aw = max(1e-6, ax2 - ax1)
        ah = max(1e-6, ay2 - ay1)
        bw = max(1e-6, bx2 - bx1)
        bh = max(1e-6, by2 - by1)

        inter_w = max(0.0, min(ax2, bx2) - max(ax1, bx1))
        inter_h = max(0.0, min(ay2, by2) - max(ay1, by1))
        inter = inter_w * inter_h

        union = (aw * ah) + (bw * bh) - inter
        iou = inter / union if union > 0 else 0.0

        horizontal_overlap_ratio = inter_w / max(1e-6, min(aw, bw))
        vertical_gap = max(0.0, max(ay1, by1) - min(ay2, by2))
        avg_h = (ah + bh) / 2.0

        # Merge if boxes overlap, or if they are strongly aligned and nearly touching.
        if iou >= 0.10:
            return True
        if horizontal_overlap_ratio >= 0.60 and vertical_gap <= (0.20 * avg_h):
            return True
        return False

    def _merge_adjacent_detections(self, detections):
        """
        Merge adjacent smoke detections that NMS can miss when boxes are touching but not overlapping.
        """
        if len(detections) <= 1:
            return detections

        merged = [dict(det) for det in detections]
        changed = True

        while changed:
            changed = False
            next_merged = []
            used = [False] * len(merged)

            for i, det_i in enumerate(merged):
                if used[i]:
                    continue

                current = {
                    'class': det_i['class'],
                    'confidence': det_i['confidence'],
                    'bbox': dict(det_i['bbox']),
                }
                used[i] = True

                for j in range(i + 1, len(merged)):
                    if used[j]:
                        continue
                    det_j = merged[j]
                    if current['class'] != det_j['class']:
                        continue
                    if not self._boxes_should_merge(current['bbox'], det_j['bbox']):
                        continue

                    current['bbox']['x1'] = min(current['bbox']['x1'], det_j['bbox']['x1'])
                    current['bbox']['y1'] = min(current['bbox']['y1'], det_j['bbox']['y1'])
                    current['bbox']['x2'] = max(current['bbox']['x2'], det_j['bbox']['x2'])
                    current['bbox']['y2'] = max(current['bbox']['y2'], det_j['bbox']['y2'])
                    current['confidence'] = max(current['confidence'], det_j['confidence'])
                    used[j] = True
                    changed = True

                next_merged.append(current)

            merged = next_merged

        merged.sort(key=lambda d: d['confidence'], reverse=True)
        return merged

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
            print(" Model not loaded. Cannot run inference.")
            return {}

        if not os.path.isdir(image_dir):
            print(f" Directory not found: {image_dir}")
            return {}

        # Get all image files
        image_extensions = ('.jpg', '.jpeg', '.png', '.bmp', '.tiff')
        image_files = [f for f in os.listdir(image_dir)
                      if f.lower().endswith(image_extensions)]

        if not image_files:
            print(f"  No images found in {image_dir}")
            return {}

        print(f" Found {len(image_files)} images. Running inference...")

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
                    'detections': self.last_detections
                }

                # Save if requested
                if save_results and annotated_img is not None:
                    output_path = os.path.join(
                        INFERENCE_RESULTS_DIR,
                        f'pred_{img_file}'
                    )
                    cv2.imwrite(output_path, annotated_img)
                    print(f"       Saved: {output_path}")

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
        print(f" Saved: {output_path}")

    def print_detections(self, results_dict):
        """
        Print detection summary for all processed images

        Args:
            results_dict: Dictionary with results from predict_batch
        """
        print("\n" + "="*60)
        print(" DETECTION SUMMARY")
        print("="*60)

        total_detections = 0
        images_with_smoke = 0

        for img_name, result_info in results_dict.items():
            detections = result_info['detections']
            num_detections = len(detections)
            total_detections += num_detections

            if num_detections > 0:
                images_with_smoke += 1
                print(f"\n {img_name}")
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
    detector = SmokeDetector(
        model_path=args.model,
        conf_threshold=args.conf,
        iou_threshold=INFERENCE_IOU
    )

    if detector.model is None:
        print("\n Cannot proceed without a model. Please provide --model argument.")
        print(f"   Example: python inference.py --model models/best.pt --image path/to/image.jpg")
        return

    # Create results directory
    os.makedirs(INFERENCE_RESULTS_DIR, exist_ok=True)

    # Run inference
    if os.path.isdir(args.image):
        print(f"\n Processing directory: {args.image}")
        results = detector.predict_batch(
            args.image,
            visualize=True,
            save_results=args.save
        )
        detector.print_detections(results)
    else:
        print(f"\n  Processing single image: {args.image}")
        results, annotated = detector.predict_single_image(args.image, visualize=True)

        if results and annotated is not None:
            detections = detector.last_detections
            print(f"\n Detections: {len(detections)}")
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
