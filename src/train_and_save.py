"""
Train and save models to disk.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import pickle
import numpy as np
from src.loader import DavisLoader
from src.features import extract_pixel_features, features_to_training_data
from src.train import train_logistic_regression, train_knn


def train_and_save():
    print("Training models and saving to disk...")
    
    loader = DavisLoader("data/DAVIS/DAVIS")
    
    # Use 5 videos for training
    train_list = loader.load_video_list("train")[:5]
    print(f"Training on: {train_list}")
    
    all_X = []
    all_y = []
    
    for video_name in train_list:
        print(f"  Processing: {video_name}")
        frames, masks = loader.load_sequence(video_name, max_frames=30)
        frames = frames[::5]  # 5 FPS
        masks = masks[::5]
        
        for idx in range(len(frames) - 2):
            features = extract_pixel_features(frames, idx)
            mask = masks[idx]
            X, y = features_to_training_data(features, mask, sample_ratio=0.1)
            if len(X) > 0:
                all_X.append(X)
                all_y.append(y)
    
    X_train = np.vstack(all_X)
    y_train = np.concatenate(all_y)
    print(f"\nTraining samples: {len(X_train)}")
    print(f"Class distribution: Background={np.sum(y_train==0)}, Foreground={np.sum(y_train==1)}")
    
    # Train
    print("\nTraining Logistic Regression...")
    lr_scaler, lr_model = train_logistic_regression(X_train, y_train)
    
    print("Training kNN...")
    knn_scaler, knn_model = train_knn(X_train, y_train, k=5)
    
    # Save
    models = {
        'lr_scaler': lr_scaler,
        'lr_model': lr_model,
        'knn_scaler': knn_scaler,
        'knn_model': knn_model,
    }
    
    with open("models.pkl", "wb") as f:
        pickle.dump(models, f)
    
    print("\nModels saved to: models.pkl")
    
    # Save training info
    with open("results/training_summary.txt", "w") as f:
        f.write("Model Training Summary\n")
        f.write("=" * 50 + "\n\n")
        f.write(f"Training videos: {train_list}\n")
        f.write(f"Training samples: {len(X_train)}\n")
        f.write(f"Background pixels: {np.sum(y_train==0)}\n")
        f.write(f"Foreground pixels: {np.sum(y_train==1)}\n")
        f.write(f"Models saved to: models.pkl\n")
    
    print("Training complete!")


if __name__ == "__main__":
    train_and_save()
