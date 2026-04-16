"""
Evaluation metrics and visualization.
"""

import cv2
import numpy as np
from pathlib import Path
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score


def compute_metrics(pred_mask: np.ndarray, true_mask: np.ndarray) -> dict:
    """
    Compute segmentation metrics.
    
    Metrics:
        - accuracy: overall pixel accuracy
        - precision: TP / (TP + FP)
        - recall: TP / (TP + FN)
        - f1: harmonic mean of precision and recall
        - iou: Jaccard Index (Intersection over Union)
        - dice: Dice coefficient (F1 for segmentation)
    
    Returns:
        Dictionary with metrics
    """
    pred_flat = pred_mask.flatten()
    true_flat = true_mask.flatten()
    
    # IoU (Jaccard Index)
    intersection = np.sum(pred_flat & true_flat)
    union = np.sum(pred_flat | true_flat)
    iou = intersection / union if union > 0 else 0
    
    # Dice coefficient
    dice = 2 * intersection / (np.sum(pred_flat) + np.sum(true_flat)) if (np.sum(pred_flat) + np.sum(true_flat)) > 0 else 0
    
    return {
        'accuracy': accuracy_score(true_flat, pred_flat),
        'precision': precision_score(true_flat, pred_flat, zero_division=0),
        'recall': recall_score(true_flat, pred_flat, zero_division=0),
        'f1': f1_score(true_flat, pred_flat, zero_division=0),
        'iou': iou,
        'dice': dice,
    }


def compute_boundary_metrics(pred_mask: np.ndarray, true_mask: np.ndarray, dilation: int = 5) -> dict:
    """
    Compute boundary-based metrics (used in DAVIS benchmark).
    
    - boundary_f: measures how well boundaries are aligned
    """
    import cv2
    
    # Dilate ground truth to create boundary region
    kernel = np.ones((dilation, dilation), np.uint8)
    true_boundary = cv2.dilate(true_mask.astype(np.uint8), kernel) - true_mask
    
    # Check if predictions fall within boundary region
    boundary_pixels = np.sum(true_boundary)
    correct_boundary = np.sum(pred_mask * true_boundary)
    
    if boundary_pixels > 0:
        boundary_f = correct_boundary / boundary_pixels
    else:
        boundary_f = 0
    
    return {
        'boundary_f': boundary_f,
    }


def visualize_segmentation(frame: np.ndarray, mask: np.ndarray, 
                           alpha: float = 0.5) -> np.ndarray:
    """
    Overlay predicted mask on original frame.
    Green = moving, original = background.
    """
    color_mask = np.zeros_like(frame)
    color_mask[mask == 1] = [0, 255, 0]
    
    result = cv2.addWeighted(frame, 1, color_mask, alpha, 0)
    
    return result


def visualize_comparison(frame: np.ndarray, pred_mask: np.ndarray, 
                         true_mask: np.ndarray) -> np.ndarray:
    """
    Create side-by-side comparison: Original, Predicted, Ground Truth.
    """
    H, W = frame.shape[:2]
    
    pred_overlay = visualize_segmentation(frame, pred_mask)
    true_overlay = visualize_segmentation(frame, true_mask)
    
    comparison = np.hstack([frame, pred_overlay, true_overlay])
    
    return comparison


def evaluate_video(models: dict, loader, video_name: str) -> dict:
    """
    Evaluate models on a single video.
    
    Returns:
        Dictionary with per-model metrics and sample frames
    """
    from src.loader import subsample_frames
    from src.features import extract_pixel_features
    from src.train import predict_segmentation
    
    frames, masks = loader.load_sequence(video_name)
    frames, masks = subsample_frames(frames, masks, target_fps=5)
    
    if not frames:
        return None
    
    results = {
        'logistic_regression': {'metrics': [], 'frames': []},
        'knn': {'metrics': [], 'frames': []},
    }
    
    for idx in range(len(frames) - 2):
        features = extract_pixel_features(frames, idx)
        true_mask = (masks[idx] > 127).astype(np.uint8)
        
        lr_mask = predict_segmentation(
            models['lr_scaler'], models['lr_model'], features
        )
        lr_metrics = compute_metrics(lr_mask, true_mask)
        lr_metrics.update(compute_boundary_metrics(lr_mask, true_mask))
        results['logistic_regression']['metrics'].append(lr_metrics)
        
        knn_mask = predict_segmentation(
            models['knn_scaler'], models['knn_model'], features
        )
        knn_metrics = compute_metrics(knn_mask, true_mask)
        knn_metrics.update(compute_boundary_metrics(knn_mask, true_mask))
        results['knn']['metrics'].append(knn_metrics)
        
        if idx % 5 == 0:
            comp_frame = visualize_comparison(frames[idx], lr_mask, true_mask)
            results['logistic_regression']['frames'].append(comp_frame)
    
    for model_name in results:
        if results[model_name]['metrics']:
            avg_metrics = {}
            metrics_to_avg = ['accuracy', 'precision', 'recall', 'f1', 'iou', 'dice', 'boundary_f']
            for key in metrics_to_avg:
                avg_metrics[key] = np.mean([m[key] for m in results[model_name]['metrics']])
            # J&F score (DAVIS metric: mean of IoU and F-measure)
            avg_metrics['j_and_f'] = (avg_metrics['iou'] + avg_metrics['boundary_f']) / 2
            results[model_name]['avg_metrics'] = avg_metrics
    
    return results


def save_results(results: dict, output_dir: str):
    """
    Save evaluation results to disk.
    """
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    
    with open(output / "metrics.txt", "w") as f:
        f.write("Motion Segmentation Results\n")
        f.write("=" * 50 + "\n\n")
        
        for video_name, video_results in results.items():
            if video_results is None:
                continue
            f.write(f"\nVideo: {video_name}\n")
            f.write("-" * 40 + "\n")
            
            for model_name, data in video_results.items():
                f.write(f"\n  {model_name.upper()}\n")
                if 'avg_metrics' in data:
                    for metric, value in data['avg_metrics'].items():
                        f.write(f"    {metric}: {value:.4f}\n")
    
    frames_dir = output / "frames"
    frames_dir.mkdir(exist_ok=True)
    
    for model_name, data in results.items():
        if data and data['frames']:
            model_frames_dir = frames_dir / model_name.replace(' ', '_')
            model_frames_dir.mkdir(exist_ok=True)
            
            for i, frame in enumerate(data['frames'][:10]):
                cv2.imwrite(str(model_frames_dir / f"frame_{i:03d}.jpg"), frame)
    
    if results:
        first_result = next((v for v in results.values() if v and v['frames']), None)
        if first_result:
            frame_size = first_result['frames'][0].shape[:2][::-1]
            writer = cv2.VideoWriter(
                str(output / "segmentation.mp4"),
                cv2.VideoWriter_fourcc(*'mp4v'),
                5,
                frame_size
            )
            
            for frame in first_result['frames']:
                writer.write(frame)
            
            writer.release()
            print(f"Saved video to {output / 'segmentation.mp4'}")
