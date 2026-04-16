"""
Fast evaluation with key metrics.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import pickle
import numpy as np
from src.loader import DavisLoader
from src.features import extract_pixel_features
from src.evaluate import compute_metrics, visualize_comparison
from src.train import predict_segmentation
import cv2


def fast_evaluate():
    print("Fast Evaluation - Key Metrics")
    print("=" * 50)
    
    # Load models
    with open("models.pkl", "rb") as f:
        models = pickle.load(f)
    
    loader = DavisLoader("data/DAVIS/DAVIS")
    
    # Test on 3 videos, 10 frames each
    val_list = loader.load_video_list("val")[:3]
    print(f"Testing on: {val_list}")
    
    output_dir = Path("results/fast_eval")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    all_results = {'lr': [], 'knn': []}
    frame_num = 0
    
    for video_name in val_list:
        frames, masks = loader.load_sequence(video_name, max_frames=50)
        frames = frames[::5]  # Every 5th frame
        masks = masks[::5]
        
        print(f"  {video_name}: {len(frames)} frames")
        
        for idx in range(min(10, len(frames) - 2)):
            features = extract_pixel_features(frames, idx)
            true_mask = (masks[idx] > 127).astype(np.uint8)
            
            # LR
            lr_mask = predict_segmentation(models['lr_scaler'], models['lr_model'], features)
            lr_metrics = compute_metrics(lr_mask, true_mask)
            all_results['lr'].append(lr_metrics)
            
            # kNN
            knn_mask = predict_segmentation(models['knn_scaler'], models['knn_model'], features)
            knn_metrics = compute_metrics(knn_mask, true_mask)
            all_results['knn'].append(knn_metrics)
            
            # Save 1 frame per video
            if idx == 0:
                comp = visualize_comparison(frames[idx], lr_mask, true_mask)
                cv2.imwrite(str(output_dir / f"{video_name}.jpg"), comp)
    
    # Average metrics
    print("\n" + "=" * 50)
    print("RESULTS (averaged over all frames)")
    print("=" * 50)
    
    for model, metrics_list in all_results.items():
        n = len(metrics_list)
        if n == 0:
            continue
        
        model_name = "Logistic Regression" if model == "lr" else "k-Nearest Neighbors"
        print(f"\n{model_name}:")
        
        avg = {k: sum(m[k] for m in metrics_list) / n for k in metrics_list[0].keys()}
        
        print(f"  Accuracy:  {avg['accuracy']:.4f}")
        print(f"  Precision: {avg['precision']:.4f}")
        print(f"  Recall:    {avg['recall']:.4f}")
        print(f"  F1 Score:  {avg['f1']:.4f}")
        print(f"  IoU:       {avg['iou']:.4f}")
        print(f"  Dice:      {avg['dice']:.4f}")
    
    # Save
    with open(output_dir / "metrics.txt", "w") as f:
        f.write("Fast Evaluation Results\n")
        f.write("=" * 50 + "\n\n")
        f.write(f"Test videos: {val_list}\n")
        f.write(f"Frames per video: 10\n\n")
        
        for model, metrics_list in all_results.items():
            n = len(metrics_list)
            if n == 0:
                continue
            model_name = "Logistic Regression" if model == "lr" else "kNN"
            f.write(f"\n{model_name}:\n")
            avg = {k: sum(m[k] for m in metrics_list) / n for k in metrics_list[0].keys()}
            for k, v in avg.items():
                f.write(f"  {k}: {v:.4f}\n")
    
    print(f"\nSaved to: {output_dir}")


if __name__ == "__main__":
    fast_evaluate()
