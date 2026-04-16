# Motion-Based Video Object Segmentation

## Final Project Report

---

## 1. Problem Statement

### What are we solving?

**Core Question:** Can temporal pixel change alone produce a usable foreground segmentation signal?

**In simpler terms:** Given a video, can we identify which pixels show moving objects by only looking at how pixels change between frames?

### The Task

```
Input:  Video frames (sequence of images over time)
Output: Binary mask for each frame (0 = background, 1 = moving object)
```

Example:

```
Frame t:    [person walking in park]
Frame t+1:  [person moved slightly right]

Our system detects: [green overlay on person] ← moving pixels
                   [no overlay on background] ← static pixels
```

---

## 2. Why This Problem Matters

### The Big Picture

Motion segmentation is a fundamental problem in computer vision because:

1. **Foundation for tracking** - Before you can track an object, you must first know where it is
2. **Autonomous vehicles** - Detect pedestrians, vehicles, obstacles in motion
3. **Video editing** - Automatic foreground/background separation
4. **Surveillance** - Detect moving objects in camera feeds
5. **Wildlife monitoring** - Detect camouflaged animals using motion cues

### Why Frame Differencing?

```
┌─────────────────────────────────────────────────────────────┐
│  Traditional Computer Vision Approach:                       │
│                                                             │
│  • Object detection needs appearance/features                │
│  • Appearance fails when objects blend into background       │
│  • Motion often betrays presence even when appearance doesn't│
│                                                             │
│  Example: A lizard blending into bark is nearly invisible    │
│           But it MOVES, so frame differencing can detect it │
└─────────────────────────────────────────────────────────────┘
```

### The Limits We Acknowledge

Our system WILL fail when:

- Object stops moving (no temporal change)
- Camera moves (everything appears to change)
- Lighting changes dramatically (appears as motion)

This is not a bug - it's the fundamental limitation of motion-only perception.

---

## 3. How The System Works

### Pipeline Overview

```
┌──────────────┐     ┌──────────────┐     ┌──────────────┐
│   VIDEO      │────▶│  FEATURE     │────▶│   MODEL      │
│   FRAMES     │     │  EXTRACTION  │     │  TRAINING    │
└──────────────┘     └──────────────┘     └──────────────┘
                                                  │
                                                  ▼
┌──────────────┐     ┌──────────────┐     ┌──────────────┐
│  EVALUATION  │◀────│   OUTPUT     │◀────│  SEGMENTATION│
│   METRICS    │     │   MASKS      │     │    PREDICT   │
└──────────────┘     └──────────────┘     └──────────────┘
```

### Step 1: Data Preparation

**Dataset:** DAVIS 2016 (Densely Annotated VIdeo Segmentation)

| Property     | Value                        |
| ------------ | ---------------------------- |
| Total videos | 50 (30 train, 20 val)        |
| Resolution   | 480p (854×480)               |
| Frame rate   | 24 FPS (subsampled to 5 FPS) |
| Annotations  | Pixel-level foreground masks |

```
Why DAVIS?
• Direct alignment: video frames → ground truth masks
• Standard benchmark: allows comparison with other methods
• Manageable size: fits on a laptop
```

### Step 2: Feature Extraction

For each pixel, we compute three features:

```
Feature 1: |I_t - I_{t+1}|
           Absolute difference between current frame and next frame
           Primary motion signal

Feature 2: |I_t - I_{t+2}|
           Difference skipping one frame
           Catches slightly slower motion

Feature 3: Local mean of Feature 1
           3×3 neighborhood average
           Reduces noise, stabilizes signal
```

### Step 3: Training

**Data:**

- Training videos: bear, bmx-bumps, boat, breakdance-flare, bus
- Training samples: 288,761 pixels
- Class distribution: 73% background, 27% foreground

**Models Used:**

#### Model 1: Logistic Regression

```
How it works:
• Learns weights: w1×F1 + w2×F2 + w3×F3 + b → probability
• If probability > 0.5 → predict "moving"
• If probability ≤ 0.5 → predict "background"

Why it's good:
• Interpretable (see what weights it learned)
• Fast to train
• Works well when features are informative
```

#### Model 2: k-Nearest Neighbors (kNN)

```
How it works:
• For each test pixel, find K nearest pixels in training data
• Vote on their labels
• K=5 in our case

Why it's good:
• Captures non-linear patterns
• No explicit "training" phase
• Works when similar pixels have similar labels
```

### Step 4: Prediction & Evaluation

For each frame in a test video:

```
1. Extract features (3 values per pixel)
2. Scale features (zero mean, unit variance)
3. Apply model to get probability per pixel
4. Threshold at 0.5 to get binary mask
5. Compare to ground truth
```

---

## 4. Evaluation Metrics

### Per-Pixel Metrics

| Metric        | Formula                  | What It Measures                            |
| ------------- | ------------------------ | ------------------------------------------- |
| **Accuracy**  | (TP+TN) / Total          | Overall correctness                         |
| **Precision** | TP / (TP+FP)             | Of predicted "moving", how many are correct |
| **Recall**    | TP / (TP+FN)             | Of actual "moving", how many we detected    |
| **F1 Score**  | 2×P×R / (P+R)            | Balance between precision and recall        |
| **IoU**       | \|A∩B\| / \|A∪B\|        | Jaccard Index - overlap measure             |
| **Dice**      | 2\|A∩B\| / (\|A\|+\|B\|) | Similar to F1, common in segmentation       |

### Boundary Metrics (DAVIS Benchmark)

| Metric         | What It Measures                                         |
| -------------- | -------------------------------------------------------- |
| **Boundary F** | How well predicted boundaries align with true boundaries |
| **J&F**        | Mean of IoU and Boundary F (DAVIS's main metric)         |

### Our Results

#### Baseline Model (Initial Implementation)

```
Model: Logistic Regression
Test Videos: blackswan, bmx-trees, breakdance, camel, car-roundabout

Metric      Value    Interpretation
────────────────────────────────────────────────────
accuracy    0.27     27% of pixels correctly classified
precision   0.06     6% of predictions are correct (many false alarms)
recall      0.66     66% of moving pixels detected
f1          0.11     Overall balance score
iou         0.06     Very low overlap with ground truth
dice        0.11     Similar to F1
```

#### Improved Model (With Post-Processing)

```
Improvements Applied:
1. Adaptive threshold in training data selection
2. Morphological post-processing (close + open + small region removal)
3. More training videos (8 vs 5)
4. More training samples (729K vs 288K)

Metric      Baseline    Improved    Change
────────────────────────────────────────────
accuracy    0.27        0.15       -44%
precision   0.06        0.10       +67%
recall      0.66        0.89       +35%
f1          0.11        0.18       +64%
iou         0.06        0.10       +67%
dice        0.11        0.18       +64%
```

Model: Logistic Regression
Test Videos: blackswan, bmx-trees, breakdance, camel, car-roundabout

Metric Value Interpretation
─────────────────────────────────────────────────────
accuracy 0.27 27% of pixels correctly classified
precision 0.06 6% of predictions are correct (many false alarms)
recall 0.66 66% of moving pixels detected
f1 0.11 Overall balance score
iou 0.06 Very low overlap with ground truth
dice 0.11 Similar to F1

```

### What This Tells Us

```

┌─────────────────────────────────────────────────────────────┐
│ High recall (66%), low precision (6%) │
│ │
│ This means: │
│ ✓ We catch most moving objects │
│ ✗ But we also flag many background pixels as moving │
│ │
│ The model is "sensitive" but not "selective" │
└─────────────────────────────────────────────────────────────┘

```

---

## 5. Project Structure

```

basic_AI/project/
│
├── data/
│ └── DAVIS/DAVIS/ # DAVIS 2016 dataset (~2GB)
│ ├── JPEGImages/480p/ # Video frames
│ ├── Annotations/480p/ # Ground truth masks
│ └── ImageSets/480p/ # Train/val splits
│
├── src/
│ ├── **init**.py
│ ├── loader.py # Data loading utilities
│ ├── features.py # Frame differencing, feature extraction
│ ├── train.py # Model training (LR, kNN)
│ ├── evaluate.py # Metrics computation
│ ├── main.py # Full pipeline
│ ├── train_and_save.py # Train + save models
│ ├── quick_test.py # Fast sanity check
│ ├── generate_results.py # Create visualizations
│ ├── fast_evaluate.py # Full evaluation
│ └── minimal_eval.py # Quick evaluation
│
├── results/
│ ├── frames/ # Visualization images
│ ├── minimal_eval/
│ │ ├── blackswan.jpg # Original | Predicted | Ground Truth
│ │ ├── bmx-trees.jpg
│ │ └── metrics.txt
│ └── summary.txt
│
├── models.pkl # Trained models (16MB)
├── requirements.txt # Python dependencies
└── README.md # Project documentation

```

---

## 6. Key Design Decisions

### Why These Choices?

| Decision                      | Why                                                                     |
| ----------------------------- | ----------------------------------------------------------------------- |
| **5 FPS subsampling**         | 24 FPS is overkill; 5 FPS still captures motion while being faster      |
| **960×540 target size**       | Original is 854×480; minimal resizing preserves quality                 |
| **10% pixel sampling**        | Full video = millions of pixels; sampling keeps training fast           |
| **class_weight='balanced'**   | Background pixels outnumber foreground 3:1; prevents bias to background |
| **threshold=25 for features** | Filters noisy low-difference pixels while keeping true motion           |

---

## 7. Limitations & Future Work

### Current Limitations

```

1. CAMERA MOTION FAILS
   If camera moves, everything appears to change
   Fix: Optical flow compensation

2. OBJECT STOPS = DISAPPEARS
   No motion = no detection
   Fix: Temporal memory, appearance features

3. NOISE FROM LIGHTING
   Brightness changes create false motion
   Fix: Normalized frame difference, adaptive thresholding

4. NO OBJECT IDENTITY
   We detect motion, not "which object"
   Fix: Instance segmentation, tracking

```

### Potential Improvements

```

Short-term (within project scope):
• Tune threshold for better precision/recall balance
• Add temporal smoothing (average predictions over 3 frames)
• Try different pixel sampling strategies

Long-term (beyond project scope):
• Add appearance features (color, texture)
• Use optical flow for camera motion compensation
• Add tracking to maintain object identity
• Use deep features from CNN

````

---

## 8. Code Summary

### Training

```python
# Load data
loader = DavisLoader("data/DAVIS/DAVIS")
train_list = loader.load_video_list("train")[:5]

# Extract features
for video in train_list:
    frames, masks = loader.load_sequence(video)
    for idx in range(len(frames) - 2):
        features = extract_pixel_features(frames, idx)
        X, y = features_to_training_data(features, masks[idx])
        all_X.append(X)
        all_y.append(y)

# Train
X_train = np.vstack(all_X)
y_train = np.concatenate(all_y)
scaler, model = train_logistic_regression(X_train, y_train)
````

### Inference

```python
# Load models
with open("models.pkl", "rb") as f:
    models = pickle.load(f)

# Predict
features = extract_pixel_features(frames, idx)
pred_mask = predict_segmentation(
    models['lr_scaler'],
    models['lr_model'],
    features
)
```

---

## 9. Conclusion

### What We Built

A minimal motion segmentation system that:

- Takes video frames as input
- Extracts temporal difference features
- Trains simple classifiers (LR, kNN)
- Outputs binary segmentation masks
- Computes standard evaluation metrics

### What We Learned

1. **Motion is a valid perceptual signal** - even simple differencing detects moving objects
2. **Trade-off between precision and recall** - high sensitivity comes at cost of false positives
3. **Frame differencing is foundational** - modern systems build on these principles

### Key Takeaway

> "Can motion alone produce a usable segmentation signal?"

**Answer: Partially yes.** Our system detects 66% of moving pixels, but with 94% false positives. It works well enough to demonstrate the principle, but real applications need additional techniques (appearance features, temporal memory, optical flow) to achieve practical performance.

---

## References

1. Perazzi et al., "DAVIS: A Benchmark Dataset and Evaluation Methodology for Video Object Segmentation," CVPR 2016
2. Bouwmans, "Background Subtraction for Video Surveillance," Pattern Recognition 2014
3. Géron, "Hands-On Machine Learning with Scikit-Learn, Keras, and TensorFlow," 2022

---

**Project Location:**

```
C:\Users\omarb\Desktop\basic_AI\project
```

**To Run:**

```bash
cd C:\Users\omarb\Desktop\basic_AI\project
python src/minimal_eval.py       # Quick test
python src/train_and_save.py     # Retrain models
python src/generate_results.py    # Create visualizations
```
