"""
Minimal evaluation - LR only, fast.
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


print("Minimal Evaluation (LR only)")
print("=" * 40)

# Load LR only
with open("models.pkl", "rb") as f:
    models = pickle.load(f)

loader = DavisLoader("data/DAVIS/DAVIS")
val_list = loader.load_video_list("val")[:2]

all_metrics = []
output_dir = Path("results/minimal_eval")
output_dir.mkdir(exist_ok=True)

for video_name in val_list:
    print(f"Testing: {video_name}")
    frames, masks = loader.load_sequence(video_name, max_frames=30)
    frames = frames[::5]
    masks = masks[::5]
    
    for idx in range(min(8, len(frames) - 2)):
        features = extract_pixel_features(frames, idx)
        true_mask = (masks[idx] > 127).astype(np.uint8)
        
        # LR prediction only
        pred_mask = predict_segmentation(models['lr_scaler'], models['lr_model'], features)
        metrics = compute_metrics(pred_mask, true_mask)
        all_metrics.append(metrics)
        
        if idx == 0:
            comp = visualize_comparison(frames[idx], pred_mask, true_mask)
            cv2.imwrite(str(output_dir / f"{video_name}.jpg"), comp)

# Average
n = len(all_metrics)
if n > 0:
    avg = {k: sum(m[k] for m in all_metrics) / n for k in all_metrics[0].keys()}
    
    print("\n" + "=" * 40)
    print("AVERAGE RESULTS")
    print("=" * 40)
    for k, v in avg.items():
        print(f"  {k:12s}: {v:.4f}")
    
    with open(output_dir / "metrics.txt", "w") as f:
        f.write(f"Test videos: {val_list}\n")
        f.write(f"Frames: {n}\n\n")
        for k, v in avg.items():
            f.write(f"{k}: {v:.4f}\n")

print("\nDone!")
