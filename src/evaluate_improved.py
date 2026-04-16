"""
Improved evaluation with post-processing and temporal smoothing.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import pickle
import numpy as np
from src.loader import DavisLoader
from src.features import extract_pixel_features, postprocess_mask, temporal_smooth
from src.evaluate import compute_metrics, visualize_comparison
from src.train import predict_segmentation
import cv2


def predict_with_postprocess(scaler, model, features, threshold=0.5):
    """Predict mask with morphological post-processing."""
    mask = predict_segmentation(scaler, model, features, threshold)
    return postprocess_mask(mask)


def evaluate_improved():
    print("=" * 60)
    print("IMPROVED MOTION SEGMENTATION EVALUATION")
    print("=" * 60)
    
    # Load models
    print("\nLoading trained models...")
    with open("models.pkl", "rb") as f:
        models = pickle.load(f)
    
    loader = DavisLoader("data/DAVIS/DAVIS")
    
    # Test configurations
    configs = [
        {"name": "Baseline (no post-process)", "postprocess": False, "threshold": 0.5},
        {"name": "Higher threshold (0.7)", "postprocess": False, "threshold": 0.7},
        {"name": "With post-process", "postprocess": True, "threshold": 0.5},
        {"name": "Post-process + high threshold", "postprocess": True, "threshold": 0.6},
    ]
    
    val_list = loader.load_video_list("val")[:3]
    print(f"\nTesting on: {val_list}")
    
    results_dir = Path("results/improved")
    results_dir.mkdir(parents=True, exist_ok=True)
    
    all_results = {cfg["name"]: [] for cfg in configs}
    
    for video_name in val_list:
        print(f"\nProcessing: {video_name}")
        frames, masks = loader.load_sequence(video_name, max_frames=50)
        frames = frames[::4]
        masks = masks[::4]
        
        # Collect raw predictions for temporal smoothing
        raw_masks = []
        for idx in range(len(frames) - 2):
            features = extract_pixel_features(frames, idx)
            raw_masks.append(predict_segmentation(
                models['lr_scaler'], models['lr_model'], features, threshold=0.5
            ))
        
        # Apply temporal smoothing
        if len(raw_masks) >= 3:
            smoothed_masks = temporal_smooth(raw_masks, window_size=3)
        else:
            smoothed_masks = raw_masks
        
        # Evaluate each configuration
        for cfg in configs:
            for idx in range(min(len(frames) - 2, 10)):
                features = extract_pixel_features(frames, idx)
                true_mask = (masks[idx] > 127).astype(np.uint8)
                
                if cfg["postprocess"]:
                    # Use smoothed mask for post-process configs
                    pred_mask = smoothed_masks[idx] if idx < len(smoothed_masks) else raw_masks[idx]
                    pred_mask = postprocess_mask(pred_mask)
                else:
                    pred_mask = predict_segmentation(
                        models['lr_scaler'], models['lr_model'], features, 
                        threshold=cfg["threshold"]
                    )
                
                metrics = compute_metrics(pred_mask, true_mask)
                all_results[cfg["name"]].append(metrics)
    
    # Print results
    print("\n" + "=" * 60)
    print("RESULTS COMPARISON")
    print("=" * 60)
    
    best_config = None
    best_f1 = 0
    
    for cfg in configs:
        metrics_list = all_results[cfg["name"]]
        if not metrics_list:
            continue
        
        n = len(metrics_list)
        avg = {k: sum(m[k] for m in metrics_list) / n for k in metrics_list[0].keys()}
        
        print(f"\n{cfg['name']}:")
        print(f"  Precision: {avg['precision']:.4f}")
        print(f"  Recall:    {avg['recall']:.4f}")
        print(f"  F1 Score:  {avg['f1']:.4f}")
        print(f"  IoU:       {avg['iou']:.4f}")
        
        if avg['f1'] > best_f1:
            best_f1 = avg['f1']
            best_config = cfg
    
    print("\n" + "=" * 60)
    print(f"BEST CONFIG: {best_config['name']}")
    print(f"F1 Score:    {best_f1:.4f}")
    print("=" * 60)
    
    # Generate visualizations with best config
    print("\nGenerating visualizations with best config...")
    vis_dir = results_dir / "best_frames"
    vis_dir.mkdir(exist_ok=True)
    
    for video_name in val_list[:2]:
        frames, masks = loader.load_sequence(video_name, max_frames=30)
        frames = frames[::4]
        masks = masks[::4]
        
        for idx in range(min(5, len(frames) - 2)):
            features = extract_pixel_features(frames, idx)
            true_mask = (masks[idx] > 127).astype(np.uint8)
            
            pred_mask = predict_segmentation(
                models['lr_scaler'], models['lr_model'], features,
                threshold=best_config["threshold"]
            )
            
            if best_config["postprocess"]:
                pred_mask = postprocess_mask(pred_mask)
            
            comp = visualize_comparison(frames[idx], pred_mask, true_mask)
            cv2.imwrite(str(vis_dir / f"{video_name}_frame_{idx}.jpg"), comp)
    
    print(f"Saved to: {vis_dir}")
    
    # Save detailed results
    with open(results_dir / "comparison.txt", "w") as f:
        f.write("Configuration Comparison Results\n")
        f.write("=" * 60 + "\n\n")
        f.write(f"Test videos: {val_list}\n\n")
        
        for cfg in configs:
            metrics_list = all_results[cfg["name"]]
            if not metrics_list:
                continue
            n = len(metrics_list)
            avg = {k: sum(m[k] for m in metrics_list) / n for k in metrics_list[0].keys()}
            
            f.write(f"{cfg['name']}:\n")
            f.write(f"  Precision: {avg['precision']:.4f}\n")
            f.write(f"  Recall:    {avg['recall']:.4f}\n")
            f.write(f"  F1 Score:  {avg['f1']:.4f}\n")
            f.write(f"  IoU:       {avg['iou']:.4f}\n\n")
    
    print(f"\nResults saved to: {results_dir}")
    return best_config


if __name__ == "__main__":
    evaluate_improved()
