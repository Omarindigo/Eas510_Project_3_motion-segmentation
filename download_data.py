"""
Download script for DAVIS 2016 dataset.

Usage:
    python download_data.py

This script downloads and extracts the DAVIS 2016 (480p) dataset.
The dataset will be placed in data/DAVIS/DAVIS/
"""

import os
import zipfile
import urllib.request
from pathlib import Path

DATASET_URL = "https://data.vision.ee.ethz.ch/csergi/share/davis/DAVIS-2016-trainval-480p.zip"
DATA_DIR = Path(__file__).parent.parent / "data" / "DAVIS"
ZIP_PATH = DATA_DIR / "davis-data.zip"

def download_dataset():
    """Download DAVIS 2016 dataset."""
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    
    if (DATA_DIR / "DAVIS" / "ImageSets").exists():
        print("DAVIS dataset already exists. Skipping download.")
        return
    
    print(f"Downloading DAVIS 2016 dataset...")
    print(f"URL: {DATASET_URL}")
    
    urllib.request.urlretrieve(DATASET_URL, ZIP_PATH)
    print(f"Downloaded to {ZIP_PATH}")
    
    print("Extracting dataset...")
    with zipfile.ZipFile(ZIP_PATH, 'r') as zip_ref:
        zip_ref.extractall(DATA_DIR)
    
    print("Extraction complete!")
    print(f"Dataset location: {DATA_DIR / 'DAVIS'}")
    
    # Clean up zip file
    ZIP_PATH.unlink()
    print("Removed zip file.")

if __name__ == "__main__":
    download_dataset()
