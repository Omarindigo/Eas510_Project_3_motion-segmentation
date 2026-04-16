# Motion Segmentation Project

Simple motion-based video object segmentation using classical computer vision and machine learning.

## Setup

```bash
pip install -r requirements.txt
```

## Download Dataset

1. Go to https://davischallenge.org/
2. Download DAVIS 2016 (480p version recommended)
3. Extract to `data/DAVIS/`

Expected structure:

```
data/DAVIS/
├── ImageSets/
│   └── 2016/
│       ├── train.txt
│       └── val.txt
├── JPEGImages/
│   └── 480p/
│       ├── bear/
│       └── ...
└── Annotations/
    └── 480p/
        ├── bear/
        └── ...
```

## Run

```bash
python src/main.py
```

## Project Structure

```
motion-segmentation/
├── src/
│   ├── loader.py     # DAVIS data loading
│   ├── features.py   # Frame differencing features
│   ├── train.py      # Model training
│   ├── evaluate.py   # Metrics and visualization
│   └── main.py       # Pipeline orchestration
├── results/          # Output (created after running)
└── requirements.txt
```

## Method

1. Extract frames from video
2. Compute pixel differences: |I*t - I*{t+1}| and |I*t - I*{t+2}|
3. Add local neighborhood smoothing
4. Train classifier (Logistic Regression or kNN)
5. Generate segmentation masks

## Expected Output

- Metrics: accuracy, precision, recall, F1 score
- Visualizations: frame overlays with segmentation
- Video: segmentation output
