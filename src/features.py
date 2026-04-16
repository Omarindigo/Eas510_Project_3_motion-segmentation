"""
Feature extraction from video frames.
"""

import cv2
import numpy as np


def compute_frame_difference(frame1: np.ndarray, frame2: np.ndarray) -> np.ndarray:
    """Compute absolute pixel difference between two frames."""
    gray1 = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)
    diff = cv2.absdiff(gray1, gray2)
    return diff


def compute_multi_frame_difference(frames: list, idx: int) -> tuple:
    """Compute frame differences at multiple offsets."""
    diff_1 = compute_frame_difference(frames[idx], frames[idx + 1]) if idx + 1 < len(frames) else None
    diff_2 = compute_frame_difference(frames[idx], frames[idx + 2]) if idx + 2 < len(frames) else None
    return diff_1, diff_2


def compute_local_mean_difference(diff: np.ndarray, kernel_size: int = 3) -> np.ndarray:
    """Compute local mean of frame difference (3×3 neighborhood)."""
    kernel = np.ones((kernel_size, kernel_size), np.float32) / (kernel_size ** 2)
    smoothed = cv2.filter2D(diff.astype(np.float32), -1, kernel)
    return smoothed.astype(np.uint8)


def extract_pixel_features(frames: list, idx: int) -> np.ndarray:
    """
    Extract feature vector for every pixel at frame idx.
    
    Features:
        - diff_1: |I_t - I_{t+1}|
        - diff_2: |I_t - I_{t+2}|
        - local_mean: smoothed diff
    """
    H, W = frames[idx].shape[:2]
    features = np.zeros((H, W, 3), dtype=np.float32)
    
    if idx + 1 < len(frames):
        features[:, :, 0] = compute_frame_difference(frames[idx], frames[idx + 1])
    
    if idx + 2 < len(frames):
        features[:, :, 1] = compute_frame_difference(frames[idx], frames[idx + 2])
    
    if features[:, :, 0].size > 0:
        features[:, :, 2] = compute_local_mean_difference(features[:, :, 0])
    
    return features


def features_to_training_data(features: np.ndarray, mask: np.ndarray, 
                               sample_ratio: float = 0.1,
                               threshold: int = None) -> tuple:
    """
    Convert feature matrix to training-ready format.
    
    Uses adaptive threshold based on video statistics.
    """
    H, W = features.shape[:2]
    
    X_flat = features.reshape(-1, 3)
    y_flat = (mask > 127).astype(np.int32).reshape(-1)
    
    # Adaptive threshold: mean + 1.5 * std
    if threshold is None:
        diff_values = X_flat[:, 0]
        threshold = np.mean(diff_values) + 1.5 * np.std(diff_values)
        threshold = int(np.clip(threshold, 15, 50))  # Keep reasonable range
    
    meaningful = X_flat[:, 0] > threshold
    clear_label = y_flat == 1
    
    mask_filter = meaningful | clear_label
    
    X_filtered = X_flat[mask_filter]
    y_filtered = y_flat[mask_filter]
    
    n_samples = int(len(X_filtered) * sample_ratio)
    if n_samples > 0:
        indices = np.random.choice(len(X_filtered), size=n_samples, replace=False)
        X = X_filtered[indices]
        y = y_filtered[indices]
    else:
        X = X_filtered
        y = y_filtered
    
    return X, y


def postprocess_mask(mask: np.ndarray, 
                    min_area: int = 100,
                    kernel_size: int = 5) -> np.ndarray:
    """
    Clean up mask with morphological operations.
    
    Steps:
        1. Morphological close (fill gaps)
        2. Morphological open (remove noise)
        3. Remove small regions
    
    Args:
        mask: Binary mask (H, W) with 0/1 values
        min_area: Minimum connected region size
        kernel_size: Size of morphological kernel
    
    Returns:
        Cleaned binary mask
    """
    mask_uint8 = mask.astype(np.uint8)
    
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
    
    # Close: fill small holes
    closed = cv2.morphologyEx(mask_uint8, cv2.MORPH_CLOSE, kernel)
    
    # Open: remove small noise
    opened = cv2.morphologyEx(closed, cv2.MORPH_OPEN, kernel)
    
    # Remove small connected components
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(opened, connectivity=8)
    
    cleaned = np.zeros_like(opened)
    for i in range(1, num_labels):  # Skip background (label 0)
        if stats[i, cv2.CC_STAT_AREA] >= min_area:
            cleaned[labels == i] = 1
    
    return cleaned


def temporal_smooth(masks: list, window_size: int = 3) -> list:
    """
    Apply temporal smoothing to a sequence of masks.
    
    Uses a simple moving average filter.
    
    Args:
        masks: List of binary masks
        window_size: Number of frames to average
    
    Returns:
        Smoothed masks
    """
    if len(masks) < window_size:
        return masks
    
    half_window = window_size // 2
    smoothed = []
    
    for i in range(len(masks)):
        start = max(0, i - half_window)
        end = min(len(masks), i + half_window + 1)
        
        # Average over window
        window_masks = masks[start:end]
        avg_mask = np.mean([m.astype(np.float32) for m in window_masks], axis=0)
        
        # Threshold at 0.5
        smoothed_mask = (avg_mask >= 0.5).astype(np.uint8)
        smoothed.append(smoothed_mask)
    
    return smoothed
