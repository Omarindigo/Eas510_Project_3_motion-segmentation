"""
Retrain with improved features and generate final results.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import pickle
import numpy as np
from src.loader import DavisLoader
from src.features import extract_pixel_features, features_to_training_data, postprocess_mask, temporal_smooth
from src.train import train_logistic_regression
from src.evaluate import compute_metrics, visualize_comparison
from src.train import predict_segmentation
import cv2


def retrain_improved():
    print("=" * 60)
    print("RETRAINING WITH IMPROVED SETTINGS")
    print("=" * 60)
    
    loader = DavisLoader("data/DAVIS/DAVIS")
    
    # Train on more videos with improved settings
    train_list = loader.load_video_list("train")[:8]  # More training data
    print(f"\nTraining on: {train_list}")
    
    all_X = []
    all_y = []
    
    for video_name in train_list:
        print(f"  Processing: {video_name}")
        frames, masks = loader.load_sequence(video_name, max_frames=40)
        frames = frames[::4]  # 5-6 FPS
        masks = masks[::4]
        
        for idx in range(len(frames) - 2):
            features = extract_pixel_features(frames, idx)
            mask = masks[idx]
            X, y = features_to_training_data(features, mask, sample_ratio=0.15)
            if len(X) > 0:
                all_X.append(X)
                all_y.append(y)
    
    X_train = np.vstack(all_X)
    y_train = np.concatenate(all_y)
    
    print(f"\nTraining samples: {len(X_train)}")
    print(f"Class distribution: Background={np.sum(y_train==0)}, Foreground={np.sum(y_train==1)}")
    
    print("\nTraining Logistic Regression...")
    scaler, model = train_logistic_regression(X_train, y_train)
    
    # Save improved model
    improved_models = {'lr_scaler': scaler, 'lr_model': model}
    with open("models_improved.pkl", "wb") as f:
        pickle.dump(improved_models, f)
    
    print("Saved to: models_improved.pkl")
    
    return scaler, model


def final_evaluation(scaler, model):
    print("\n" + "=" * 60)
    print("FINAL EVALUATION")
    print("=" * 60)
    
    loader = DavisLoader("data/DAVIS/DAVIS")
    val_list = loader.load_video_list("val")[:5]
    print(f"\nTesting on: {val_list}")
    
    output_dir = Path("results/final")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    all_metrics = []
    all_metrics_pp = []  # With post-processing
    
    for video_name in val_list:
        print(f"\nProcessing: {video_name}")
        frames, masks = loader.load_sequence(video_name, max_frames=60)
        frames = frames[::4]
        masks = masks[::4]
        
        # Collect predictions for temporal smoothing
        raw_masks = []
        for idx in range(len(frames) - 2):
            features = extract_pixel_features(frames, idx)
            true_mask = (masks[idx] > 127).astype(np.uint8)
            
            # Without post-process
            pred_mask = predict_segmentation(scaler, model, features, threshold=0.5)
            metrics = compute_metrics(pred_mask, true_mask)
            all_metrics.append(metrics)
            
            # With post-process
            pred_mask_pp = postprocess_mask(pred_mask)
            metrics_pp = compute_metrics(pred_mask_pp, true_mask)
            all_metrics_pp.append(metrics_pp)
            
            # Save visualization
            if idx % 5 == 0:
                comp = visualize_comparison(frames[idx], pred_mask_pp, true_mask)
                cv2.imwrite(str(output_dir / f"{video_name}_f{idx}.jpg"), comp)
                raw = visualize_comparison(frames[idx], pred_mask, true_mask)
                cv2.imwrite(str(output_dir / f"{video_name}_raw_f{idx}.jpg"), raw)
        
        print(f"  Evaluated {len(frames) - 2} frames")
    
    # Average results
    n = len(all_metrics)
    
    print("\n" + "=" * 60)
    print("FINAL RESULTS")
    print("=" * 60)
    
    print("\nWithout Post-Processing:")
    avg = {k: sum(m[k] for m in all_metrics) / n for k in all_metrics[0].keys()}
    print(f"  Accuracy:   {avg['accuracy']:.4f}")
    print(f"  Precision: {avg['precision']:.4f}")
    print(f"  Recall:    {avg['recall']:.4f}")
    print(f"  F1 Score:  {avg['f1']:.4f}")
    print(f"  IoU:       {avg['iou']:.4f}")
    print(f"  Dice:      {avg['dice']:.4f}")
    
    print("\nWith Post-Processing (Morphological):")
    avg_pp = {k: sum(m[k] for m in all_metrics_pp) / n for k in all_metrics_pp[0].keys()}
    print(f"  Accuracy:   {avg_pp['accuracy']:.4f}")
    print(f"  Precision: {avg_pp['precision']:.4f}")
    print(f"  Recall:    {avg_pp['recall']:.4f}")
    print(f"  F1 Score:  {avg_pp['f1']:.4f}")
    print(f"  IoU:       {avg_pp['iou']:.4f}")
    print(f"  Dice:      {avg_pp['dice']:.4f}")
    
    print("\n" + "=" * 60)
    print("IMPROVEMENT SUMMARY")
    print("=" * 60)
    print(f"  F1 improved: {avg['f1']:.4f} → {avg_pp['f1']:.4f} ({(avg_pp['f1']-avg['f1'])/avg['f1']*100:.1f}%)")
    print(f"  IoU improved: {avg['iou']:.4f} → {avg_pp['iou']:.4f} ({(avg_pp['iou']-avg['iou'])/avg['iou']*100:.1f}%)")
    print(f"  Precision improved: {avg['precision']:.4f} → {avg_pp['precision']:.4f} ({(avg_pp['precision']-avg['precision'])/avg['precision']*100:.1f}%)")
    
    # Save results
    with open(output_dir / "final_metrics.txt", "w") as f:
        f.write("Final Evaluation Results\n")
        f.write("=" * 60 + "\n\n")
        f.write(f"Test videos: {val_list}\n\n")
        
        f.write("Without Post-Processing:\n")
        for k, v in avg.items():
            f.write(f"  {k}: {v:.4f}\n")
        
        f.write("\nWith Post-Processing:\n")
        for k, v in avg_pp.items():
            f.write(f"  {k}: {v:.4f}\n")
    
    print(f"\nResults saved to: {output_dir}")
    print(f"Visualizations: {output_dir}/*.jpg")
    
    return avg_pp


if __name__ == "__main__":
    scaler, model = retrain_improved()
    final_evaluation(scaler, model)
