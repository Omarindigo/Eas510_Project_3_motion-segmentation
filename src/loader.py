"""
DAVIS Dataset Loader
Handles loading video frames and ground truth masks.
"""

import cv2
import numpy as np
from pathlib import Path


class DavisLoader:
    """Load DAVIS 2016 video sequences and annotations."""
    
    def __init__(self, davis_root: str):
        self.root = Path(davis_root)
        self.frames_dir = self.root / "JPEGImages" / "480p"
        self.masks_dir = self.root / "Annotations" / "480p"
        
    def load_video_list(self, split: str = "train") -> list[str]:
        """Load list of video names from split."""
        split_file = self.root / "ImageSets" / "480p" / f"{split}.txt"
        if not split_file.exists():
            print(f"Warning: Split file not found: {split_file}")
            return []
        with open(split_file) as f:
            lines = [line.strip() for line in f]
        # Extract unique video names from paths
        # Format: /JPEGImages/480p/bear/00000.jpg
        video_names = set()
        for line in lines:
            parts = line.split('/')
            if len(parts) >= 4:
                video_names.add(parts[3])  # video_name is at index 3
        return sorted(list(video_names))
    
    def load_sequence(self, video_name: str, max_frames: int = None):
        """
        Load all frames and masks for a video sequence.
        
        Returns:
            frames: List of numpy arrays (H, W, 3) BGR
            masks: List of numpy arrays (H, W) with 0=background, 255=foreground
        """
        frames = []
        masks = []
        
        frame_dir = self.frames_dir / video_name
        if not frame_dir.exists():
            print(f"Warning: Video directory not found: {frame_dir}")
            return frames, masks
        
        frame_files = sorted(frame_dir.glob("*.jpg"))
        
        if max_frames:
            frame_files = frame_files[:max_frames]
        
        for f in frame_files:
            frame = cv2.imread(str(f))
            if frame is None:
                continue
            frames.append(frame)
            
            mask_path = self.masks_dir / video_name / (f.stem + ".png")
            if mask_path.exists():
                mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
                masks.append(mask)
            else:
                masks.append(np.zeros(frame.shape[:2], dtype=np.uint8))
        
        return frames, masks
    
    def get_frame_pair(self, frames: list, idx: int, offset: int = 1):
        """
        Get frame pair for differencing.
        
        Args:
            frames: List of frames
            idx: Current frame index
            offset: How many frames ahead to compare (1 = next frame)
        
        Returns:
            (frame_t, frame_t_offset) tuple
        """
        if idx + offset >= len(frames):
            return None, None
        return frames[idx], frames[idx + offset]


def preprocess_frame(frame: np.ndarray, target_size: tuple = (540, 960)) -> np.ndarray:
    """
    Preprocess frame for the pipeline.
    
    - Resize to target dimensions
    """
    if frame.shape[:2] != target_size:
        frame = cv2.resize(frame, (target_size[1], target_size[0]))
    return frame


def subsample_frames(frames: list, masks: list, target_fps: int = 5, original_fps: int = 24) -> tuple:
    """
    Subsample frames and masks to target FPS.
    """
    step = max(1, original_fps // target_fps)
    subsampled_frames = frames[::step]
    subsampled_masks = masks[::step] if masks else [np.zeros_like(frames[0])]
    return subsampled_frames, subsampled_masks
