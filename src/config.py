"""
Configuration settings for motion segmentation project.
"""

import numpy as np
import random

RANDOM_SEED = 42

np.random.seed(RANDOM_SEED)
random.seed(RANDOM_SEED)

from pathlib import Path

DATA_DIR = Path(__file__).parent.parent / "data" / "DAVIS" / "DAVIS"

TRAIN_VIDEOS = 8
TEST_VIDEOS = 5
MAX_FRAMES_PER_VIDEO = 40
FRAME_SKIP = 4
TARGET_FPS = 5

SAMPLE_RATIO = 0.1

ADAPTIVE_THRESHOLD = True
BASE_THRESHOLD = None

POSTPROCESS_ENABLED = True
KERNEL_SIZE = 9  # Morphological kernel size
MIN_AREA = 300   # Min connected region size

CLASSIFICATION_THRESHOLD = 0.5  # Not used - using optical flow instead

MODEL_TYPE = "logistic_regression"
KNN_K = 5

OUTPUT_DIR = "results"
