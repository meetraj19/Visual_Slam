import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
from glob import glob

# Function to compute dense optical flow using Farneback method
def compute_optical_flow(frame1, frame2):
    # Convert frames to grayscale
    gray1 = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)

    # Compute dense optical flow using Farneback method
    flow = cv2.calcOpticalFlowFarneback(gray1, gray2, None, 0.5, 3, 15, 3, 5, 1.2, 0)

    return flow

# Function to visualize the optical flow
def visualize_optical_flow(frame, flow):
    h, w = frame.shape[:2]
    # Create mesh grid
    y, x = np.mgrid[0:h:10, 0:w:10].reshape(2, -1).astype(int)
    fx, fy = flow[y, x].T

    # Create an image to draw the flow
    vis = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    # Draw lines and circles
    for (x0, y0), (dx, dy) in zip(np.column_stack((x, y)), np.column_stack((fx, fy))):
        cv2.arrowedLine(vis, (x0, y0), (int(x0+dx), int(y0+dy)), (0, 255, 0), 1, tipLength=0.5)
        cv2.circle(vis, (x0, y0), 1, (0, 255, 0), -1)

    return vis

# Function to process a directory of images
def process_directory(directory_path):
    # Get all image file paths from the directory
    image_paths = sorted(glob(os.path.join(directory_path, "*.png")))

    # Ensure there are at least two images
    if len(image_paths) < 2:
        print("Need at least two images to compute optical flow.")
        return

    # Iterate over pairs of consecutive images
    for i in range(len(image_paths) - 1):
        frame1 = cv2.imread(image_paths[i])
        frame2 = cv2.imread(image_paths[i + 1])

        # Compute optical flow
        flow = compute_optical_flow(frame1, frame2)

        # Visualize the optical flow
        vis = visualize_optical_flow(frame1, flow)

        # Display the result
        plt.figure(figsize=(10, 10))
        plt.title(f'Optical Flow between {os.path.basename(image_paths[i])} and {os.path.basename(image_paths[i + 1])}')
        plt.imshow(vis)
        plt.axis('off')
        plt.show()

# Path to the directory containing images
directory_path = '/home/thales1/VSLAM/a/pythonProject1/DATA/Data2'

# Process the directory
process_directory(directory_path)
