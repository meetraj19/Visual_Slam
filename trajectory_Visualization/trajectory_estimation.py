import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import json

# Function to load global map from CSV file
def load_global_map(filename):
    return np.loadtxt(filename, delimiter=',')

# Function to load global poses from JSON file
def load_global_poses(filename):
    with open(filename, 'r') as f:
        data = json.load(f)
    global_poses = [np.array(pose) for pose in data['global_poses']]
    return global_poses

# Function to visualize global map using matplotlib (2D)
def visualize_map_2d(global_map):
    plt.figure()
    plt.scatter(global_map[:, 0], global_map[:, 1], s=1)
    plt.title('Global Map 2D')
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.show()

# Function to visualize global map using matplotlib 3D
def visualize_map_3d(global_map):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(global_map[:, 0], global_map[:, 1], global_map[:, 2], s=1, c='g')
    ax.set_title('Global Map 3D')
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    plt.show()

# Function to visualize global poses using matplotlib 3D
def visualize_poses_3d(global_poses):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    for pose in global_poses:
        position = pose[:3, 3]
        ax.scatter(position[0], position[1], position[2], c='r', s=50)
    ax.set_title('Global Poses 3D')
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    plt.show()

# Function to visualize camera trajectory using matplotlib 3D
def visualize_camera_trajectory(global_poses):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    trajectory = np.array([pose[:3, 3] for pose in global_poses])
    ax.plot(trajectory[:, 0], trajectory[:, 1], trajectory[:, 2], 'b-', marker='o', markersize=5, label='Camera Trajectory')
    ax.scatter(trajectory[:, 0], trajectory[:, 1], trajectory[:, 2], c='r', s=50, label='Camera Positions')
    ax.set_title('Camera Trajectory 3D')
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.legend()
    plt.show()

# Main function to load and visualize global map, poses, and camera trajectory
def main():
    # Load global map
    global_map_file = 'global_map.csv'
    global_map = load_global_map(global_map_file)

    # Visualize global map
    visualize_map_2d(global_map)
    visualize_map_3d(global_map)

    # Load and visualize global poses
    global_poses_file = 'global_poses.json'
    global_poses = load_global_poses(global_poses_file)
    visualize_poses_3d(global_poses)

    # Visualize camera trajectory
    visualize_camera_trajectory(global_poses)

if __name__ == '__main__':
    main()
