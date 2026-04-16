"""
Main script for Motion-Based Video Object Segmentation.

This script implements a complete pipeline for detecting moving objects in video
using frame differencing and machine learning classifiers.

Usage:
    python main.py

Output:
    - Trained model (models_final.pkl)
    - Evaluation metrics (printed to console)
    - Visualization images (results/frames/)
    - Segmentation video (results/segmentation.mp4)

Random seed is set for reproducibility.
"""

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import cv2
import pickle
import numpy as np
import random
from pathlib import Path

from config import (
    RANDOM_SEED, DATA_DIR, TRAIN_VIDEOS, TEST_VIDEOS,
    MAX_FRAMES_PER_VIDEO, FRAME_SKIP, SAMPLE_RATIO,
    POSTPROCESS_ENABLED, KERNEL_SIZE, MIN_AREA,
    CLASSIFICATION_THRESHOLD, MODEL_TYPE, KNN_K,
    OUTPUT_DIR
)

from loader import DavisLoader
from features import (
    extract_pixel_features, features_to_training_data, 
    postprocess_mask, temporal_smooth
)
from train import train_logistic_regression, train_knn, predict_segmentation
from evaluate import compute_metrics, visualize_comparison


np.random.seed(RANDOM_SEED)
random.seed(RANDOM_SEED)


def train_model(loader, train_list):
    """Train models on training videos."""
    print("\n" + "=" * 60)
    print("TRAINING")
    print("=" * 60)
    
    all_X = []
    all_y = []
    
    for video_name in train_list:
        print(f"  Processing: {video_name}")
        frames, masks = loader.load_sequence(video_name, max_frames=MAX_FRAMES_PER_VIDEO)
        
        if not frames:
            continue
        
        frames = frames[::FRAME_SKIP]
        masks = masks[::FRAME_SKIP]
        
        for idx in range(len(frames) - 2):
            features = extract_pixel_features(frames, idx)
            mask = masks[idx]
            X, y = features_to_training_data(features, mask, sample_ratio=SAMPLE_RATIO)
            if len(X) > 0:
                all_X.append(X)
                all_y.append(y)
    
    if not all_X:
        print("Error: No training data!")
        return None, None
    
    X_train = np.vstack(all_X)
    y_train = np.concatenate(all_y)
    
    print(f"\nTraining samples: {len(X_train)}")
    print(f"Background pixels: {np.sum(y_train == 0)}")
    print(f"Foreground pixels: {np.sum(y_train == 1)}")
    
    print(f"\nTraining {MODEL_TYPE}...")
    scaler, model = train_logistic_regression(X_train, y_train)
    
    models = {'scaler': scaler, 'model': model}
    
    with open("models_final.pkl", "wb") as f:
        pickle.dump(models, f)
    
    print("Model saved to: models_final.pkl")
    
    return scaler, model


def evaluate_model(loader, scaler, model, test_list):
    """Evaluate model on test videos."""
    print("\n" + "=" * 60)
    print("EVALUATION")
    print("=" * 60)
    
    output_dir = Path(OUTPUT_DIR)
    frames_dir = output_dir / "frames"
    frames_dir.mkdir(parents=True, exist_ok=True)
    
    all_metrics = []
    all_masks = []
    
    for video_name in test_list:
        print(f"\nProcessing: {video_name}")
        frames, masks = loader.load_sequence(video_name, max_frames=50)
        
        if not frames:
            continue
        
        frames = frames[::FRAME_SKIP]
        masks = masks[::FRAME_SKIP]
        
        video_masks = []
        
        for idx in range(len(frames) - 2):
            features = extract_pixel_features(frames, idx)
            true_mask = (masks[idx] > 127).astype(np.uint8)
            
            pred_mask = predict_segmentation(scaler, model, features, 
                                          threshold=CLASSIFICATION_THRESHOLD)
            
            if POSTPROCESS_ENABLED:
                pred_mask = postprocess_mask(pred_mask, 
                                          min_area=MIN_AREA,
                                          kernel_size=KERNEL_SIZE)
            
            metrics = compute_metrics(pred_mask, true_mask)
            all_metrics.append(metrics)
            video_masks.append(pred_mask)
            
            if idx % 5 == 0:
                comp = visualize_comparison(frames[idx], pred_mask, true_mask)
                cv2.imwrite(str(frames_dir / f"{video_name}_f{idx}.jpg"), comp)
        
        all_masks.extend(video_masks[:10])
        
        print(f"  Evaluated {len(frames) - 2} frames")
    
    if not all_metrics:
        print("Error: No metrics computed!")
        return
    
    n = len(all_metrics)
    
    avg = {k: sum(m[k] for m in all_metrics) / n for k in all_metrics[0].keys()}
    
    print("\n" + "=" * 60)
    print("FINAL RESULTS")
    print("=" * 60)
    print(f"\nMetrics (averaged over {n} frames):")
    print(f"  Accuracy:   {avg['accuracy']:.4f}")
    print(f"  Precision:  {avg['precision']:.4f}")
    print(f"  Recall:    {avg['recall']:.4f}")
    print(f"  F1 Score: {avg['f1']:.4f}")
    print(f"  IoU:       {avg['iou']:.4f}")
    print(f"  Dice:      {avg['dice']:.4f}")
    
    generate_video(frames[:min(10, len(frames))], all_masks[:10], output_dir / "segmentation.mp4")
    
    return avg


def generate_video(frames, masks, output_path):
    """Generate video with segmentation overlay."""
    if not frames or not masks:
        return
    
    h, w = frames[0].shape[:2]
    
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    writer = cv2.VideoWriter(str(output_path), fourcc, 5, (w, h))
    
    for i, (frame, mask) in enumerate(zip(frames, masks)):
        overlay = visualize_comparison(frame, mask, mask)
        writer.write(overlay)
    
    writer.release()
    print(f"\nVideo saved to: {output_path}")


def main():
    print("=" * 60)
    print("MOTION-BASED VIDEO OBJECT SEGMENTATION")
    print("=" * 60)
    print(f"\nRandom seed: {RANDOM_SEED}")
    print(f"Data directory: {DATA_DIR}")
    print(f"Training videos: {TRAIN_VIDEOS}")
    print(f"Test videos: {TEST_VIDEOS}")
    print(f"Post-processing: {POSTPROCESS_ENABLED}")
    
    loader = DavisLoader(DATA_DIR)
    
    train_list = loader.load_video_list("train")[:TRAIN_VIDEOS]
    test_list = loader.load_video_list("val")[:TEST_VIDEOS]
    
    print(f"\nTraining on: {train_list}")
    print(f"Testing on: {test_list}")
    
    scaler, model = train_model(loader, train_list)
    
    if scaler is None:
        print("Training failed!")
        return
    
    results = evaluate_model(loader, scaler, model, test_list)
    
    print("\n" + "=" * 60)
    print("DONE")
    print("=" * 60)
    print(f"\nOutput files:")
    print(f"  - models_final.pkl (trained model)")
    print(f"  - results/frames/*.jpg (visualizations)")
    print(f"  - results/segmentation.mp4 (video)")


if __name__ == "__main__":
    main()
