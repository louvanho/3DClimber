"""
:module: mergeDepthAndSeg
:synopsis: Script that merges depth and segmentation data to produce a combined dataset.

.. code-block:: console

   Usage:
      python mergeDepthAndSeg.py <input_pkl> <input_depth_video> <input_seg_video> <input_depth_pkl> <output_pkl> <mergeMode: multiperson, premiere> <center: 0 average, 1 head> <display: 0 No, 1 Yes>

:param inputPklName: Path to input pkl file with annotations.
:param videoInDepthName: Path to input depth video file.
:param videoInMaskName: Path to input segmentation video file.
:param videoInDepthPklName: Path to additional input pkl for depth metadata.
:param outputPklName: Path to output pkl file with merged data.
:param mergeMode: The merge mode, "multiperson" or "premiere".
:param depthCenterMode: Center detection mode: 0 for average, 1 for head-based.
:param display: 0 or 1 to disable/enable intermediate visual display.

:author: (your name)
:date: 2023-10
"""

import cv2
import sys
import pickle
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import mode  
from scipy.stats import median_abs_deviation
from premiere.functionsMoge import colorizeDepthImage
from premiere.functionsDepth import unpackDepthImage
from tqdm import tqdm

# Check for proper argument count
if len(sys.argv) != 9:
    print("Usage: python mergeDepthAndSeg.py <input_pkl> <input_depth_video> <input_seg_video> <input_depth_pkl> <output_pkl> <mergeMode: multiperson, premiere> <center: 0 average, 1 head> <display: 0 No, 1 Yes>")
    sys.exit(1)

# Parse command-line arguments
inputPklName = sys.argv[1]
videoInDepthName = sys.argv[2]
videoInMaskName = sys.argv[3]
videoInDepthPklName = sys.argv[4]
outputPklName = sys.argv[5]
mergeMode = sys.argv[6]
depthCenterMode = int(sys.argv[7])
display = int(sys.argv[8]) == 1

premiereMergeMode = (mergeMode == "premiere")

# Open video captures
videoDepth = cv2.VideoCapture(videoInDepthName)
videoMask = cv2.VideoCapture(videoInMaskName)

# Read input pkl with extra data
print("Read input pkl: ", inputPklName)
file = open(inputPklName, 'rb')
dataPKL = pickle.load(file)
file.close()

# Read additional pkl for depth info
with open(videoInDepthPklName, 'rb') as file:
    videoDepthPKL = pickle.load(file)

# Retrieve video properties
width = int(videoDepth.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(videoDepth.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = int(videoDepth.get(cv2.CAP_PROP_FPS))
frames_count = int(videoDepth.get(cv2.CAP_PROP_FRAME_COUNT))

# Check if videos opened properly
if videoDepth.isOpened() == False:
    print('[!] error opening the depth video')
    sys.exit(1)
if videoMask.isOpened() == False:
    print('[!] error opening the mask video')
    sys.exit(1)

# If display mode is on, prepare Matplotlib
if display:
    plt.ion()
    fig, ax = plt.subplots(figsize=(10, 6))

allFrameData = []
frameCount = 0

# Print chosen center mode
if depthCenterMode == 0:
    print("average")
else:
    print("head")

pbar = tqdm(total=frames_count, unit=' frames', dynamic_ncols=True, position=0, leave=True)

# Main loop over frames
while videoDepth.isOpened():
    retDepth, frameDepth = videoDepth.read()
    retMask, frameMask = videoMask.read()

    # End loop if frames cannot be read
    if not retDepth or not retMask:
        break

    # Max ID in segmentation frame
    max_value_in_frameMask = np.max(frameMask)

    # Retrieve min/max depth from videoDepthPKL for current frame
    min_depth = videoDepthPKL[frameCount][0]
    max_depth = videoDepthPKL[frameCount][1]

    # Convert raw depth frame to visible scale
    depth, mask = unpackDepthImage(frameDepth, min_depth, max_depth, brg2rgb=True)

    # If display, create a colorized depth for visualization
    if display:
        visuDepth = colorizeDepthImage(depth)

    # If using average center mode, erode segmentation mask
    if depthCenterMode == 0:
        kernel = np.ones((5, 5), np.uint8)
        erodedMask = cv2.erode(frameMask, kernel, iterations=1)
        maskSeg = (erodedMask == 0)
        if display:
            visuDepth[maskSeg] = 0
    else:
        # If using head-based center, skip erosion
        erodedMask = frameMask
        maskSeg = (frameMask == 0)
        if display:
            visuDepth[maskSeg] = 0

    depth_values_list = []
    centers_list = []
    frameData = []

    # Loop through each object ID in the segmentation
    for i in range(1, max_value_in_frameMask + 1):
        if depthCenterMode == 0:
            # Erosion-based average center
            maskSegId = erodedMask[:, :, 0] == i
            depth_values_in_maskSeg = depth[maskSegId]
            depth_values_list.append(depth_values_in_maskSeg)

            if depth_values_in_maskSeg.size > 0:
                # Calculate median absolute deviation and filter outliers
                mad = median_abs_deviation(depth_values_in_maskSeg, nan_policy='omit')
                median = np.nanmedian(depth_values_in_maskSeg)
                filtered_values = depth_values_in_maskSeg[np.abs(depth_values_in_maskSeg - median) <= 2 * mad]
                mean = np.nanmean(filtered_values)
                center = mean
                centers_list.append(center)
                data = [i - 1, center, depth_values_in_maskSeg.size]
            else:
                centers_list.append(np.nan)
                data = [i - 1, np.nan, 0]

            frameData.append(data)
        else:
            # Head-based center mode uses data in dataPKL
            humans = dataPKL['allFrameHumans'][frameCount]
            data = [i - 1, -10, -1]
            for pos in range(len(humans)):
                human = humans[pos]
                if premiereMergeMode:
                    # "premiere" mode uses index for matching
                    if pos == (i - 1):
                        loc = human['loc']
                        x1, y1 = loc[0], loc[1]
                        if 0 <= int(y1) < depth.shape[0] and 0 <= int(x1) < depth.shape[1]:
                            n = 5
                            half_n = n // 2
                            y_start = max(0, int(y1) - half_n)
                            y_end = min(depth.shape[0], int(y1) + half_n + 1)
                            x_start = max(0, int(x1) - half_n)
                            x_end = min(depth.shape[1], int(x1) + half_n + 1)
                            depth_window = depth[y_start:y_end, x_start:x_end]
                            center = depth_window.mean()
                            if display:
                                center_coordinates = (int(x1), int(y1))
                                radius = 5
                                color = (0, 0, 255)
                                thickness = -1
                                cv2.circle(visuDepth, center_coordinates, radius, color, thickness)
                                centers_list.append(center)
                        else:
                            center = -10
                            if display:
                                centers_list.append(center)
                        data = [humans[i - 1]['id'], center, -1]
                else:
                    # "multiperson" mode matches by idsam
                    if human['idsam'] == (i - 1):
                        loc = human['loc']
                        x1, y1 = loc[0], loc[1]
                        if 0 <= int(y1) < depth.shape[0] and 0 <= int(x1) < depth.shape[1]:
                            n = 5
                            half_n = n // 2
                            y_start = max(0, int(y1) - half_n)
                            y_end = min(depth.shape[0], int(y1) + half_n + 1)
                            x_start = max(0, int(x1) - half_n)
                            x_end = min(depth.shape[1], int(x1) + half_n + 1)
                            depth_window = depth[y_start:y_end, x_start:x_end]
                            center = depth_window.mean()
                            if display:
                                center_coordinates = (int(x1), int(y1))
                                radius = 5
                                color = (0, 0, 255)
                                thickness = -1
                                cv2.circle(visuDepth, center_coordinates, radius, color, thickness)
                                centers_list.append(center)
                        else:
                            center = -10
                            if display:
                                centers_list.append(center)
                        data = [i - 1, center, -1]
                        break
            frameData.append(data)

    # If display is enabled, show updated depth
    if display:
        cv2.imshow('Depth Frame', visuDepth)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    allFrameData.append(frameData)

    # Optionally show real-time plot of computed centers
    if display:
        ax.clear()
        for i, center in enumerate(centers_list, start=1):
            if not np.isnan(center):
                ax.axvline(center * 2, color=f"C{i%10}", linestyle='dashed', linewidth=2, label=f'Mask {i} Center')
        ax.set_xlabel('Depth Values')
        ax.set_ylabel('Frequency')
        ax.set_title('Center for each Mask ID')
        ax.set_xlim(0, 5)
        ax.legend()
        plt.draw()
        plt.pause(0.001)

    frameCount += 1
    pbar.update(1)

pbar.close()

# Write merged data to output pkl
with open(outputPklName, 'wb') as handle:
    pickle.dump(allFrameData, handle, protocol=pickle.HIGHEST_PROTOCOL)

# Clean up
videoDepth.release()
videoMask.release()
cv2.destroyAllWindows()