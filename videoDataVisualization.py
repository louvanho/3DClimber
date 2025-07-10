"""
Video Data Visualization with Pose and Segmentation Overlays.

This script visualizes human pose data (from a pickle file) and segmentation masks
(from a segmentation video) overlaid on a video. The combined result is displayed
frame by frame with color-coded segmentation regions and 2D joint keypoints.

**Usage**:
    1. Specify paths directly:
        python videoDataVisualization.py <video_path> <input_pkl_path> <video_seg_path>
    2. Use a directory containing the necessary files:
        python videoDataVisualization.py <directory>
        - Expects:
            - video.mp4
            - nlf-clean.pkl
            - nlf-videoSegmentation.mp4

:param video_path: Path to the input video file.
:param input_pkl_path: Path to the pickle file containing pose data.
:param video_seg_path: Path to the segmentation video file.
"""

import sys
import cv2
import pickle
import random
import os
import numpy as np
from tqdm import tqdm

# Validate command-line arguments
if (len(sys.argv) != 3) and (len(sys.argv) != 4):
    print("Usage: python videoDataVisualization.py <video_path> <input_pkl_path> <video_seg_path>")
    print("or")
    print("Usage: python videoDataVisualization.py <directory> <0: final pkl, 1: final filtered>")
    sys.exit(1)

# Set paths based on input arguments
if len(sys.argv) == 3:
    # Assume input directory structure
    directory = sys.argv[1]
    filtered = int(sys.argv[2])==1
    if filtered:
        input_pkl_path = os.path.join(directory, "nlf-final-filtered.pkl")
    else:
        input_pkl_path = os.path.join(directory, "nlf-final.pkl")
    video_path = os.path.join(directory, "video.mp4")
    video_seg_path = os.path.join(directory, "nlf-videoSegmentation.mp4")
else:
    video_path = sys.argv[1]
    input_pkl_path = sys.argv[2]
    video_seg_path = sys.argv[3]

print("Video Path:", video_path)
print("Input PKL Path:", input_pkl_path)
print("Video Segmentation Path:", video_seg_path)

# Open video and segmentation video
video = cv2.VideoCapture(video_path)
videoSeg = cv2.VideoCapture(video_seg_path)

# Retrieve video properties
fps = video.get(cv2.CAP_PROP_FPS)
width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
total_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))

# Load pose data from pickle file
print("Reading pose pickle:", input_pkl_path)
with open(input_pkl_path, 'rb') as file:
    dataPKL = pickle.load(file)

allFrameHumans = dataPKL['allFrameHumans']

# Generate a fixed set of random colors for segmentation visualization
random.seed(42)  # Ensure reproducibility
colors = [(random.randint(0, 255), random.randint(0, 255), random.randint(0, 255)) for _ in range(200)]

# Initialize a progress bar
pbar = tqdm(total=total_frames, unit=' frames', dynamic_ncols=True, position=0, leave=True)
count = 0

# Process video frames
while video.isOpened():
    ret1, frame = video.read()
    ret2, frameSeg = videoSeg.read()

    # Stop if no more frames or pose data
    if not ret1 or not ret2 or count >= len(allFrameHumans):
        break

    # Initialize a blank segmentation visualization frame
    frameSegVisu = np.zeros((height, width, 3), dtype=np.uint8)

    # Extract unique IDs from the segmentation frame
    unique_ids = np.unique(frameSeg[:, :, 0])
    max_id = unique_ids.max()

    # Create a lookup table (LUT) for segmentation ID colors
    lut = np.zeros((max_id + 1, 3), dtype=np.uint8)
    for seg_id in range(1, max_id + 1):
        lut[seg_id] = colors[(seg_id - 1) % len(colors)]

    # Apply the LUT to the segmentation frame
    mask = frameSeg[:, :, 0] == 0  # Mask for background
    frameSegVisu[mask] = (0, 0, 0)  # Set background to black
    frameSegVisu[~mask] = lut[frameSeg[~mask][:, 0]]  # Set object regions to their colors

    # Combine the original frame with the segmentation visualization
    alpha = 0.5  # Transparency factor for blending
    combined_frame = cv2.addWeighted(frame, alpha, frameSegVisu, 1 - alpha, 0)

    # Overlay 2D joint keypoints from pose data
    humans = allFrameHumans[count]
    for human in humans:
        color = (255, 255, 255)  # White color for keypoints
        for i in range(len(human['j2d_smplx'])):
            x, y = human['j2d_smplx'][i]
            cv2.circle(combined_frame, (int(x), int(y)), 1, color, -1)

    # Display the combined frame
    cv2.imshow('frame', combined_frame)
    cv2.waitKey(1)
    count += 1
    pbar.update(1)

# Release resources
pbar.close()
video.release()
videoSeg.release()
cv2.destroyAllWindows()
