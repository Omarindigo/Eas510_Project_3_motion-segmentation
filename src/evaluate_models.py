"""
Comprehensive evaluation with all DAVIS-style metrics.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import pickle
import numpy as np
from src.loader import DavisLoader
from src.features import extract_pixel_features
from src.evaluate import compute_metrics, compute_boundary_metrics, visualize_comparison
from src.train import predict_segmentation
import cv2


def evaluate_saved_models():
    print("=" * 60)
    print("Motion Segmentation - Comprehensive Evaluation")
    print("=" * 60)
    
    # Load trained models
    print("\nLoading trained models...")
    with open("models.pkl", "rb") as f:
        models = pickle.load(f)
    print("Models loaded successfully!")
    
    loader = DavisLoader("data/DAVIS/DAVIS")
    
    # Test on validation videos
    val_list = loader.load_video_list("val")[:5]
    print(f"\nEvaluating on: {val_list}")
    
    output_dir = Path("results/evaluation")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    all_results = {
        'logistic_regression': [],
        'knn': [],
    }
    
    frame_count = 0
    
    for video_name in val_list:
        print(f"\nProcessing: {video_name}")
        frames, masks = loader.load_sequence(video_name, max_frames=40)
        
        if not frames:
            continue
        
        # Subsample
        frames = frames[::4]
        masks = masks[::4]
        
        for idx in range(len(frames) - 2):
            features = extract_pixel_features(frames, idx)
            true_mask = (masks[idx] > 127).astype(np.uint8)
            
            # Logistic Regression
            lr_mask = predict_segmentation(
                models['lr_scaler'], models['lr_model'], features
            )
            lr_metrics = compute_metrics(lr_mask, true_mask)
            lr_metrics.update(compute_boundary_metrics(lr_mask, true_mask))
            all_results['logistic_regression'].append(lr_metrics)
            
            # kNN
            knn_mask = predict_segmentation(
                models['knn_scaler'], models['knn_model'], features
            )
            knn_metrics = compute_metrics(knn_mask, true_mask)
            knn_metrics.update(compute_boundary_metrics(knn_mask, true_mask))
            all_results['knn'].append(knn_metrics)
            
            # Save visualization every 5th frame
            if idx % 5 == 0 and frame_count < 15:
                comp = visualize_comparison(frames[idx], lr_mask, true_mask)
                cv2.imwrite(str(output_dir / f"frame_{frame_count:02d}.jpg"), comp)
                frame_count += 1
        
        print(f"  Processed {len(frames) - 2} frames")
    
    # Compute averages
    print("\n" + "=" * 60)
    print("FINAL RESULTS")
    print("=" * 60)
    
    metrics_to_show = ['accuracy', 'precision', 'recall', 'f1', 'iou', 'dice', 'boundary_f', 'j_and_f']
    
    for model_name, metrics_list in all_results.items():
        if not metrics_list:
            continue
        
        print(f"\n{model_name.upper().replace('_', ' ')}:")
        print("-" * 40)
        
        avg = {}
        for metric in metrics_to_show[:-1]:  # Exclude j_and_f from individual calculation
            avg[metric] = sum(m[metric] for m in metrics_list) / len(metrics_list)
        
        # J&F (DAVIS metric)
        avg['j_and_f'] = (avg['iou'] + avg['boundary_f']) / 2
        
        for metric in metrics_to_show:
            value = avg.get(metric, 0)
            print(f"  {metric:15s}: {value:.4f}")
    
    # Save to file
    with open(output_dir / "metrics.txt", "w") as f:
        f.write("Motion Segmentation - Comprehensive Evaluation\n")
        f.write("=" * 60 + "\n\n")
        f.write(f"Test videos: {val_list}\n")
        f.write(f"Total frames evaluated: {frame_count}\n\n")
        
        for model_name, metrics_list in all_results.items():
            f.write(f"\n{model_name.upper().replace('_', ' ')}:\n")
            f.write("-" * 40 + "\n")
            
            avg = {}
            for metric in metrics_to_show[:-1]:
                avg[metric] = sum(m[metric] for m in metrics_list) / len(metrics_list)
            avg['j_and_f'] = (avg['iou'] + avg['boundary_f']) / 2
            
            for metric in metrics_to_show:
                value = avg.get(metric, 0)
                f.write(f"  {metric:15s}: {value:.4f}\n")
        
        f.write("\n" + "=" * 60 + "\n")
        f.write("METRIC DESCRIPTIONS:\n")
        f.write("-" * 40 + "\n")
        f.write("  accuracy  : Overall pixel accuracy\n")
        f.write("  precision : TP / (TP + FP)\n")
        f.write("  recall   : TP / (TP + FN)\n")
        f.write("  f1       : Harmonic mean of precision/recall\n")
        f.write("  iou      : Jaccard Index (Intersection over Union)\n")
        f.write("  dice     : Dice coefficient (similar to F1)\n")
        f.write("  boundary_f: Boundary-based F-measure\n")
        f.write("  j_and_f  : Mean of IoU and boundary F (DAVIS metric)\n")
    
    print(f"\n\nResults saved to: {output_dir / 'metrics.txt'}")
    print(f"Visualizations saved to: {output_dir / 'frames'}")
    print("\nDone!")


if __name__ == "__main__":
    evaluate_saved_models()
