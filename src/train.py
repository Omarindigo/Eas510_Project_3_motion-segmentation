"""
Model training for motion segmentation.
"""

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler


def train_logistic_regression(X: np.ndarray, y: np.ndarray) -> tuple:
    """
    Train logistic regression model.
    
    Returns:
        (scaler, model) - fitted scaler and model
    """
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    model = LogisticRegression(max_iter=1000, class_weight='balanced')
    model.fit(X_scaled, y)
    
    return scaler, model


def train_knn(X: np.ndarray, y: np.ndarray, k: int = 5) -> tuple:
    """
    Train k-Nearest Neighbors model.
    
    Returns:
        (scaler, model) - fitted scaler and model
    """
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    model = KNeighborsClassifier(n_neighbors=k, weights='distance')
    model.fit(X_scaled, y)
    
    return scaler, model


def predict_segmentation(scaler, model, features: np.ndarray, threshold: float = 0.5) -> np.ndarray:
    """
    Generate segmentation mask from features.
    
    Returns:
        Binary mask (H, W) with 0=background, 1=moving
    """
    H, W = features.shape[:2]
    
    X = features.reshape(-1, 3)
    X_scaled = scaler.transform(X)
    
    if hasattr(model, 'predict_proba'):
        probs = model.predict_proba(X_scaled)[:, 1]
        mask_flat = (probs > threshold).astype(np.uint8)
    else:
        mask_flat = model.predict(X_scaled).astype(np.uint8)
    
    mask = mask_flat.reshape(H, W)
    
    return mask


def train_on_videos(loader, video_names: list, sample_ratio: float = 0.1) -> dict:
    """
    Train models on multiple videos.
    
    Returns:
        Dictionary with scalers and models
    """
    from src.loader import subsample_frames
    from src.features import extract_pixel_features, features_to_training_data
    
    all_X = []
    all_y = []
    
    for video_name in video_names:
        print(f"Processing: {video_name}")
        frames, masks = loader.load_sequence(video_name)
        
        if not frames:
            print(f"  Warning: No frames loaded for {video_name}")
            continue
        
        frames, masks = subsample_frames(frames, masks, target_fps=5)
        
        for idx in range(len(frames) - 2):
            features = extract_pixel_features(frames, idx)
            mask = masks[idx]
            
            X, y = features_to_training_data(features, mask, sample_ratio=sample_ratio)
            if len(X) > 0:
                all_X.append(X)
                all_y.append(y)
    
    if not all_X:
        print("Error: No training data loaded!")
        return None
    
    X_train = np.vstack(all_X)
    y_train = np.concatenate(all_y)
    
    print(f"Total training samples: {len(X_train)}")
    print(f"Class distribution: {np.bincount(y_train)}")
    
    lr_scaler, lr_model = train_logistic_regression(X_train, y_train)
    knn_scaler, knn_model = train_knn(X_train, y_train, k=5)
    
    return {
        'lr_scaler': lr_scaler,
        'lr_model': lr_model,
        'knn_scaler': knn_scaler,
        'knn_model': knn_model,
    }
