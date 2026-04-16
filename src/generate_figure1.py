"""
Generate Figure 1 for report: Feature extraction pipeline demonstration.
Shows synthetic frames with a moving object and the resulting difference features.
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path


def create_synthetic_frames(width=200, height=150, object_radius=15, motion_pixels=8):
    """
    Create synthetic frames with a moving dark circle.
    
    Returns 3 frames where the circle moves from left to right.
    """
    frames = []
    
    for i in range(3):
        frame = np.ones((height, width), dtype=np.uint8) * 128
        x = 50 + i * motion_pixels
        y = height // 2
        
        cv2.circle(frame, (x, y), object_radius, 200, -1)
        
        frames.append(frame)
    
    return frames


def compute_frame_difference(frame1, frame2):
    """Compute absolute difference between two frames."""
    return cv2.absdiff(frame1, frame2)


def compute_local_mean(diff, kernel_size=3):
    """Compute local mean of frame difference."""
    kernel = np.ones((kernel_size, kernel_size), np.float32) / (kernel_size ** 2)
    smoothed = cv2.filter2D(diff.astype(np.float32), -1, kernel)
    return smoothed.astype(np.uint8)


def generate_figure():
    """Generate the feature extraction pipeline figure."""
    
    frames = create_synthetic_frames()
    
    diff_01 = compute_frame_difference(frames[0], frames[1])
    diff_02 = compute_frame_difference(frames[0], frames[2])
    smoothed = compute_local_mean(diff_01)
    
    fig, axes = plt.subplots(2, 3, figsize=(12, 6))
    
    titles_top = ['Frame t', 'Frame t+1', 'Frame t+2']
    titles_bottom = ['|I_t - I_{t+1}|', '|I_t - I_{t+2}|', '3x3 Smoothed']
    
    for i, ax in enumerate(axes[0]):
        ax.imshow(frames[i], cmap='gray')
        ax.set_title(titles_top[i], fontsize=12, fontweight='bold')
        ax.axis('off')
    
    axes[0][0].set_ylabel('Input Frames', fontsize=11, fontweight='bold')
    
    axes[1][0].imshow(diff_01, cmap='gray', vmin=0, vmax=255)
    axes[1][0].set_title(titles_bottom[0], fontsize=11)
    axes[1][0].axis('off')
    
    axes[1][1].imshow(diff_02, cmap='gray', vmin=0, vmax=255)
    axes[1][1].set_title(titles_bottom[1], fontsize=11)
    axes[1][1].axis('off')
    
    axes[1][2].imshow(smoothed, cmap='gray', vmin=0, vmax=255)
    axes[1][2].set_title(titles_bottom[2], fontsize=11)
    axes[1][2].axis('off')
    
    axes[1][0].set_ylabel('Feature Maps', fontsize=11, fontweight='bold')
    
    plt.tight_layout()
    
    output_path = Path(__file__).parent.parent / "results" / "figure1_feature_extraction.png"
    plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='white')
    print(f"Figure saved to: {output_path}")
    
    plt.show()


def generate_figure_with_arrows():
    """Generate figure with arrows showing the pipeline flow."""
    
    frames = create_synthetic_frames()
    
    diff_01 = compute_frame_difference(frames[0], frames[1])
    diff_02 = compute_frame_difference(frames[0], frames[2])
    smoothed = compute_local_mean(diff_01)
    
    fig = plt.figure(figsize=(14, 7))
    
    gs = fig.add_gridspec(2, 7, width_ratios=[1, 1, 1, 0.4, 1, 1, 1], wspace=0.1)
    
    ax1 = fig.add_subplot(gs[0, 0])
    ax2 = fig.add_subplot(gs[0, 1])
    ax3 = fig.add_subplot(gs[0, 2])
    ax4 = fig.add_subplot(gs[1, 0])
    ax5 = fig.add_subplot(gs[1, 1])
    ax6 = fig.add_subplot(gs[1, 2])
    
    for ax, frame, title in zip([ax1, ax2, ax3], frames, ['Frame t', 'Frame t+1', 'Frame t+2']):
        ax.imshow(frame, cmap='gray')
        ax.set_title(title, fontsize=11, fontweight='bold')
        ax.axis('off')
    
    for ax, diff, title in zip([ax4, ax5, ax6], 
                                [diff_01, diff_02, smoothed],
                                ['|I_t - I_{t+1}|', '|I_t - I_{t+2}|', '3x3 Smoothed']):
        ax.imshow(diff, cmap='gray', vmin=0, vmax=255)
        ax.set_title(title, fontsize=10)
        ax.axis('off')
    
    ax1.set_ylabel('Input Frames', fontsize=11, fontweight='bold', rotation=0, ha='right', va='center')
    ax4.set_ylabel('Feature Maps', fontsize=11, fontweight='bold', rotation=0, ha='right', va='center')
    
    fig.text(0.5, 0.02, 'Bright pixels indicate motion (large frame difference)', 
              ha='center', fontsize=10, style='italic')
    
    plt.tight_layout()
    
    output_path = Path(__file__).parent.parent / "results" / "figure1_feature_extraction.png"
    plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='white')
    print(f"Figure saved to: {output_path}")
    
    return output_path


if __name__ == "__main__":
    generate_figure_with_arrows()
