# Motion-Based Video Object Segmentation

A classical computer vision and machine learning approach to detecting and segmenting moving objects in video sequences. Developed for **EAS 510: The Basics of AI** (Spring 2026, Johns Hopkins University).

## Overview

This project implements a complete video object segmentation pipeline that combines frame differencing, optical flow, and background subtraction with machine learning classifiers to detect moving objects. Evaluated on the **DAVIS 2016** dataset.

## Quick Start

### Installation

```bash
pip install -r requirements.txt
```

### Dataset Setup

1. Download DAVIS 2016 (480p) from [davischallenge.org](https://davischallenge.org/)
2. Extract to `data/DAVIS/` with this structure:

```
data/DAVIS/
└── DAVIS/
    ├── ImageSets/480p/
    │   ├── train.txt
    │   └── val.txt
    ├── JPEGImages/480p/
    │   └── <video_name>/
    │       └── *.jpg
    └── Annotations/480p/
        └── <video_name>/
            └── *.png
```

### Run

```bash
python src/main.py
```

This trains the models, evaluates on test videos, prints metrics to console, saves trained models, and generates visualization outputs.

## Pipeline Architecture

```
┌──────────────┐     ┌──────────────────┐     ┌───────────────────┐     ┌──────────────┐     ┌───────────────────┐
│  Video Input │────>│ Background Model │────>│  Feature Extract  │────>│  Classifier   │────>│  Post-Processing  │
│  (DAVIS)     │     │ (Temporal Median)│     │  (Optical Flow +  │     │  (LR / kNN)   │     │  (Morphology +    │
└──────────────┘     └──────────────────┘     │   BG Subtraction) │     └──────────────┘     │   CC Filtering)   │
                                              └───────────────────┘                          └───────────────────┘
```

### Step 1: Background Modeling

A **temporal median** is computed over a sliding window of frames to estimate the static background:

- **Window size**: 30 frames (last 30 frames in sequence)
- **Computation**: `bg_median = median(frames[-30:])` per-pixel across all channels
- **Purpose**: Models the static scene so moving objects can be detected via deviation

### Step 2: Feature Extraction

Three features are extracted per pixel, forming a 3-dimensional feature vector:

| Feature Index | Feature | Formula | Description |
|---|---|---|---|
| 0 | `bg_diff` | `|I_t - bg_median|` | Absolute difference from background model |
| 1 | `flow_mag` | `|optical_flow| × 10` | Dense optical flow magnitude (Farneback), scaled by 10 |
| 2 | `local_mean` | `conv2D(bg_diff, 3×3 mean kernel)` | 3×3 neighborhood-averaged background difference |

#### Optical Flow Configuration (Farneback)

| Parameter | Value | Description |
|---|---|---|
| `pyr_scale` | 0.5 | Pyramid downsampling ratio |
| `levels` | 3 | Number of pyramid layers |
| `winsize` | 15 | Averaging window size |
| `iterations` | 3 | Iterations per pyramid level |
| `poly_n` | 5 | Pixel neighborhood size for polynomial fitting |
| `poly_sigma` | 1.2 | Gaussian standard deviation for polynomial fitting |

#### Adaptive Threshold for Training Data Filtering

Training samples are filtered using a video-specific adaptive threshold:

```python
threshold = mean(bg_diff) + 2.5 * std(bg_diff)
threshold = clip(threshold, 20, 60)
```

Only pixels where `bg_diff > threshold` OR `ground_truth == 1` are kept, reducing the dataset to meaningful samples.

#### Sampling

- **Sample ratio**: 10% of filtered pixels are randomly sampled for training
- **Purpose**: Reduces memory and training time while preserving class distribution

### Step 3: ML Classifiers

Two classifiers are trained on pixel-level features:

#### Logistic Regression

| Hyperparameter | Value | Description |
|---|---|---|
| `max_iter` | 1000 | Maximum optimization iterations |
| `class_weight` | `balanced` | Automatic class imbalance correction |
| `solver` | `lbfgs` (default) | Optimization algorithm |
| `scaler` | `StandardScaler` | Z-score normalization (fit on training data) |

#### k-Nearest Neighbors

| Hyperparameter | Value | Description |
|---|---|---|
| `n_neighbors` | 5 | Number of neighbors for voting |
| `weights` | `distance` | Inverse-distance weighting for votes |
| `scaler` | `StandardScaler` | Z-score normalization (fit on training data) |

#### Prediction

- Models use `predict_proba()` when available for soft predictions
- **Classification threshold**: 0.5 (probability > 0.5 → foreground)
- Predictions are reshaped from flat array back to `(H, W)` mask

### Step 4: Post-Processing

Three sequential operations clean the raw prediction mask:

| Operation | Kernel | Purpose |
|---|---|---|
| Morphological Close | 9×9 ellipse | Fill small holes/gaps within detected regions |
| Morphological Open | 9×9 ellipse | Remove isolated noise pixels |
| Connected Component Filter | min_area=300 | Remove regions smaller than 300 pixels |

### Step 5: Temporal Smoothing (Optional)

A moving average filter smooths predictions across consecutive frames:

- **Window size**: 3 frames
- **Method**: Average binary masks in window, threshold at 0.5
- **Purpose**: Reduces flickering and jitter in predictions

## Evaluation Metrics

| Metric | Formula | Description |
|---|---|---|
| **Accuracy** | `(TP + TN) / (TP + TN + FP + FN)` | Overall pixel-level correctness |
| **Precision** | `TP / (TP + FP)` | How many predicted foreground pixels are correct |
| **Recall** | `TP / (TP + FN)` | How many true foreground pixels are detected |
| **F1 Score** | `2 × (P × R) / (P + R)` | Harmonic mean of precision and recall |
| **IoU (Jaccard)** | `TP / (TP + FP + FN)` | Intersection over Union |
| **Dice** | `2 × TP / (2 × TP + FP + FN)` | Dice coefficient (F1 for segmentation) |
| **Boundary F** | `correct_boundary / boundary_pixels` | Measures boundary alignment (DAVIS benchmark) |
| **J&F** | `(IoU + Boundary_F) / 2` | DAVIS official combined score |

## Hyperparameters

All hyperparameters are defined in `src/config.py`:

| Parameter | Value | Description |
|---|---|---|
| `RANDOM_SEED` | 42 | Reproducibility (numpy, random, sklearn) |
| `TRAIN_VIDEOS` | 8 | Number of training videos |
| `TEST_VIDEOS` | 5 | Number of test videos |
| `MAX_FRAMES_PER_VIDEO` | 40 | Max frames loaded per video |
| `FRAME_SKIP` | 4 | Sample every 4th frame |
| `TARGET_FPS` | 5 | Target frame rate after subsampling |
| `SAMPLE_RATIO` | 0.1 | Fraction of filtered pixels used for training |
| `KERNEL_SIZE` | 9 | Morphological operation kernel (ellipse) |
| `MIN_AREA` | 300 | Minimum connected component size (pixels) |
| `KNN_K` | 5 | Number of neighbors for kNN |
| `CLASSIFICATION_THRESHOLD` | 0.5 | Probability threshold for foreground |
| `BG_THRESHOLD` | 15 | Background difference threshold (rule-based) |
| `FLOW_THRESHOLD` | 1.5 | Optical flow threshold (rule-based) |
| `POSTPROCESS_ENABLED` | True | Enable morphological post-processing |

## Project Structure

```
motion-segmentation/
├── src/
│   ├── __init__.py           # Package marker
│   ├── config.py             # All hyperparameters and settings
│   ├── loader.py             # DAVIS dataset loader and preprocessing
│   ├── features.py           # Feature extraction (bg subtraction, optical flow, post-processing)
│   ├── train.py              # ML model training (Logistic Regression, kNN)
│   ├── evaluate.py           # Metrics, visualization, result saving
│   └── main.py               # Pipeline entry point
├── data/
│   └── DAVIS/                # Dataset (download separately, gitignored)
├── results/                  # Generated outputs (gitignored)
│   ├── frames/               # Sample comparison frames
│   ├── segmentation.mp4      # Output video with overlay
│   └── metrics.txt           # Evaluation results
├── requirements.txt          # Python dependencies with versions
└── README.md                 # This file
```

## Results

| Method | F1 | IoU | Precision | Recall |
|---|---|---|---|---|
| Frame Differencing + ML | 0.18 | 0.10 | 0.10 | 0.89 |
| **Optical Flow + Background** | **0.27** | **0.17** | **0.18** | **0.79** |

**Key Finding:** The optical flow + background subtraction approach traded 10% recall for an 80% precision improvement, yielding the best overall F1 score. Pixel-level ML classifiers underperformed simple thresholding, suggesting that spatial context is essential for effective motion-based segmentation.

## Dependencies

| Package | Version | Purpose |
|---|---|---|
| `opencv-python` | 4.8.0.74 | Image/video I/O, optical flow, morphological operations |
| `numpy` | 1.26.3 | Array operations, random sampling, statistics |
| `scikit-learn` | 1.3.2 | Logistic Regression, kNN, StandardScaler, metrics |
| `matplotlib` | 3.8.2 | Visualization and plotting |

## Reproducibility

- `RANDOM_SEED = 42` is set globally for `numpy` and `random`
- All sklearn classifiers use `random_state=42` where applicable
- `np.random.seed()` and `random.seed()` are called at module load in `config.py`
- Running `python src/main.py` produces identical results across runs on the same hardware and dataset
