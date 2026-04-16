"""
Generate visualization output.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import cv2
import numpy as np
from src.loader import DavisLoader
from src.features import extract_pixel_features, features_to_training_data
from src.train import train_logistic_regression, predict_segmentation
from src.evaluate import visualize_comparison


def generate_visualization():
    print("Motion Segmentation - Generating Visualization")
    print("=" * 50)
    
    loader = DavisLoader("data/DAVIS/DAVIS")
    
    # Training on 3 videos
    train_list = loader.load_video_list("train")[:3]
    print(f"Training on: {train_list}")
    
    all_X = []
    all_y = []
    
    for video_name in train_list:
        frames, masks = loader.load_sequence(video_name, max_frames=25)
        frames = frames[::5]
        masks = masks[::5]
        
        for idx in range(min(len(frames) - 2, 8)):
            features = extract_pixel_features(frames, idx)
            mask = masks[idx]
            X, y = features_to_training_data(features, mask, sample_ratio=0.05)
            if len(X) > 0:
                all_X.append(X)
                all_y.append(y)
    
    X_train = np.vstack(all_X)
    y_train = np.concatenate(all_y)
    print(f"Training samples: {len(X_train)}")
    
    print("Training model...")
    scaler, model = train_logistic_regression(X_train, y_train)
    
    # Test and save frames
    test_list = loader.load_video_list("val")[:2]
    print(f"\nGenerating frames for: {test_list}")
    
    output_dir = Path("results/frames")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    for video_name in test_list:
        frames, masks = loader.load_sequence(video_name, max_frames=30)
        frames = frames[::4]  # Every 4th frame at ~5 FPS
        masks = masks[::4]
        
        for idx in range(min(len(frames) - 2, 10)):
            features = extract_pixel_features(frames, idx)
            true_mask = (masks[idx] > 127).astype(np.uint8)
            pred_mask = predict_segmentation(scaler, model, features)
            
            # Create comparison: Original | Predicted | Ground Truth
            comparison = visualize_comparison(frames[idx], pred_mask, true_mask)
            
            # Resize for smaller file size
            comparison = cv2.resize(comparison, (960, 200))
            
            # Save
            filename = f"{video_name}_frame_{idx:02d}.jpg"
            cv2.imwrite(str(output_dir / filename), comparison)
            
        print(f"  Saved 10 frames for: {video_name}")
    
    print(f"\nSaved to: {output_dir}")
    
    # Save summary
    with open("results/summary.txt", "w") as f:
        f.write("Motion Segmentation Results\n")
        f.write("=" * 50 + "\n\n")
        f.write("Method: Frame differencing + Logistic Regression\n")
        f.write("Features: |I_t - I_{t+1}|, |I_t - I_{t+2}|, local mean\n")
        f.write(f"Training videos: {train_list}\n")
        f.write(f"Test videos: {test_list}\n")
        f.write(f"Training samples: {len(X_train)}\n\n")
        f.write("Output: results/frames/*.jpg\n")
        f.write("Each image shows: Original | Predicted Mask | Ground Truth\n")
    
    print("\nDone! Check results/ folder")


if __name__ == "__main__":
    generate_visualization()
