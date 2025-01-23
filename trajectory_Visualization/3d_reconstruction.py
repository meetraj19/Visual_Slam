import numpy as np
import json
import cv2
import os
from glob import glob
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Function to load global map from CSV file
def load_global_map(filename):
    return np.loadtxt(filename, delimiter=',')

# Function to load global poses from JSON file
def load_global_poses(filename):
    with open(filename, 'r') as f:
        data = json.load(f)
    global_poses = [np.array(pose) for pose in data['global_poses']]
    return global_poses

# Function to load depth maps from a directory
def load_depth_maps(depth_maps_dir):
    depth_map_files = sorted(glob(os.path.join(depth_maps_dir, "*.npy")))
    depth_maps = [np.load(file) for file in depth_map_files]
    return depth_maps

# Function to project 2D points to 3D
def project_2d_to_3d(points_2d, depth_map, pose, K):
    points_3d = []
    for pt in points_2d:
        z = depth_map[int(pt[1]), int(pt[0])]
        if z > 0:  # Ensure valid depth value
            x = (pt[0] - K[0, 2]) * z / K[0, 0]
            y = (pt[1] - K[1, 2]) * z / K[1, 1]
            point_3d = np.array([x, y, z, 1.0])
            point_3d_global = pose @ point_3d
            points_3d.append(point_3d_global[:3])
    return points_3d

# Function to visualize the 3D reconstruction using matplotlib
def visualize_3d_reconstruction(points_3d):
    points_3d = np.array(points_3d)
    fig = plt.figure(figsize=(10, 7))
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(points_3d[:, 0], points_3d[:, 1], points_3d[:, 2], s=1, c='g')
    ax.set_title('3D Reconstruction')
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    plt.show()

# Main function for 3D reconstruction
def main():
    # Load global map and poses
    global_map_file = 'global_map.csv'
    global_poses_file = 'global_poses.json'
    depth_maps_dir = '/home/thales1/VSLAM/a/pythonProject1/DATA/DepthMaps'

    global_map = load_global_map(global_map_file)
    global_poses = load_global_poses(global_poses_file)
    depth_maps = load_depth_maps(depth_maps_dir)

    # Camera intrinsic matrix (example values, adjust to your camera)
    K = np.array([[718.856, 0, 607.1928],
                  [0, 718.856, 185.2157],
                  [0, 0, 1]])

    all_points_3d = []

    # For each frame, project 2D points to 3D
    for i, (pose, depth_map) in enumerate(zip(global_poses, depth_maps)):
        # Print some depth map statistics for debugging
        print(f"Depth map {i}: min={depth_map.min()}, max={depth_map.max()}, mean={depth_map.mean()}")

        # Generate a grid of 2D points
        h, w = depth_map.shape
        y, x = np.mgrid[0:h:10, 0:w:10].reshape(2, -1).astype(int)
        points_2d = np.vstack((x, y)).T

        # Project 2D points to 3D
        points_3d = project_2d_to_3d(points_2d, depth_map, pose, K)
        all_points_3d.extend(points_3d)

    # Visualize the 3D reconstruction
    visualize_3d_reconstruction(all_points_3d)

if __name__ == '__main__':
    main()
