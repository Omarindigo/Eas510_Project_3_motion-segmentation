"""
Quick test script - runs fast with reduced parameters.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import cv2
import numpy as np
from src.loader import DavisLoader
from src.features import extract_pixel_features, features_to_training_data
from src.train import train_logistic_regression, train_knn, predict_segmentation
from src.evaluate import compute_metrics


def quick_test():
    print("Loading data...")
    loader = DavisLoader("data/DAVIS/DAVIS")
    
    # Use only 2 videos for fast testing
    train_list = loader.load_video_list("train")[:2]
    val_list = loader.load_video_list("val")[:1]
    
    print(f"Training on: {train_list}")
    print(f"Testing on: {val_list}")
    
    # Collect training data (limit frames per video)
    all_X = []
    all_y = []
    
    for video_name in train_list:
        print(f"Processing: {video_name}")
        frames, masks = loader.load_sequence(video_name, max_frames=20)
        
        # Subsample to 5 FPS
        frames = frames[::5]
        masks = masks[::5]
        
        for idx in range(min(len(frames) - 2, 5)):
            features = extract_pixel_features(frames, idx)
            mask = masks[idx]
            X, y = features_to_training_data(features, mask, sample_ratio=0.05)
            if len(X) > 0:
                all_X.append(X)
                all_y.append(y)
    
    X_train = np.vstack(all_X)
    y_train = np.concatenate(all_y)
    
    print(f"Training samples: {len(X_train)}")
    print(f"Class distribution: {np.bincount(y_train)}")
    
    # Train models
    print("\nTraining Logistic Regression...")
    lr_scaler, lr_model = train_logistic_regression(X_train, y_train)
    
    print("Training kNN...")
    knn_scaler, knn_model = train_knn(X_train, y_train, k=3)
    
    # Test on val video
    print(f"\nTesting on: {val_list[0]}")
    frames, masks = loader.load_sequence(val_list[0], max_frames=20)
    frames = frames[::5]
    masks = masks[::5]
    
    for idx in range(min(len(frames) - 2, 3)):
        features = extract_pixel_features(frames, idx)
        true_mask = (masks[idx] > 127).astype(np.uint8)
        
        lr_mask = predict_segmentation(lr_scaler, lr_model, features)
        lr_metrics = compute_metrics(lr_mask, true_mask)
        
        knn_mask = predict_segmentation(knn_scaler, knn_model, features)
        knn_metrics = compute_metrics(knn_mask, true_mask)
        
        print(f"\nFrame {idx}:")
        print(f"  Logistic Regression: F1={lr_metrics['f1']:.3f}, Prec={lr_metrics['precision']:.3f}, Rec={lr_metrics['recall']:.3f}")
        print(f"  kNN:                F1={knn_metrics['f1']:.3f}, Prec={knn_metrics['precision']:.3f}, Rec={knn_metrics['recall']:.3f}")
    
    print("\nDone!")


if __name__ == "__main__":
    quick_test()
