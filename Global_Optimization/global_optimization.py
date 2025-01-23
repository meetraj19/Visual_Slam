import numpy as np
import json
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation as R
import cv2

# Global optimization using scipy.optimize.least_squares
def global_optimization(poses, loop_closures):
    def residuals(params):
        residuals = []
        for i in range(len(poses)):
            R_vec, t = poses[i]
            R_matrix = R.from_rotvec(params[6 * i: 6 * i + 3]).as_matrix()
            t_vec = params[6 * i + 3: 6 * i + 6]
            if i < len(poses) - 1:
                R_next = R.from_rotvec(params[6 * (i + 1): 6 * (i + 1) + 3]).as_matrix()
                t_next = params[6 * (i + 1) + 3: 6 * (i + 1) + 6]
                residuals.append(np.linalg.norm(R_matrix @ R.from_rotvec(R_vec).as_matrix() - R_next))
                residuals.append(np.linalg.norm(t_vec + t - t_next))
        for idx1, idx2 in loop_closures:
            if idx1 < len(poses) and idx2 < len(poses):
                R1 = R.from_rotvec(params[6 * idx1: 6 * idx1 + 3]).as_matrix()
                t1 = params[6 * idx1 + 3: 6 * idx1 + 6]
                R2 = R.from_rotvec(params[6 * idx2: 6 * idx2 + 3]).as_matrix()
                t2 = params[6 * idx2 + 3: 6 * idx2 + 6]
                residuals.append(np.linalg.norm(R1 - R2))
                residuals.append(np.linalg.norm(t1 - t2))
        return np.array(residuals).flatten()

    initial_params = np.zeros(6 * len(poses))
    for i, (R_vec, t) in enumerate(poses):
        initial_params[6 * i: 6 * i + 3] = R_vec
        initial_params[6 * i + 3: 6 * i + 6] = t.flatten()  # Ensure t is flattened

    result = least_squares(residuals, initial_params)
    optimized_params = result.x
    optimized_poses = [(R.from_rotvec(optimized_params[6 * i: 6 * i + 3]).as_matrix(), optimized_params[6 * i + 3: 6 * i + 6]) for i in range(len(poses))]
    return optimized_poses

# Function to load results from local optimization
def load_local_optimization_results(results_file):
    with open(results_file, 'r') as f:
        data = json.load(f)
    poses = [(R.from_matrix(np.array(pose["R_merged"])).as_rotvec(), np.array(pose["t_merged"])) for pose in data["poses"]]
    image_paths = data["image_paths"]
    return poses, image_paths

# Function to load results from loop closure detection
def load_loop_closure_results(results_file):
    with open(results_file, 'r') as f:
        data = json.load(f)
    loop_closures = data["loop_closures"]
    return loop_closures

# Function to compute global poses from relative poses
def compute_global_poses(optimized_poses):
    global_poses = []
    current_pose = np.eye(4)
    global_poses.append(current_pose)

    for R_mat, t in optimized_poses:
        transformation = np.eye(4)
        transformation[:3, :3] = R_mat
        transformation[:3, 3] = t

        current_pose = current_pose @ transformation
        global_poses.append(current_pose)

    return global_poses

# Function to save global poses to a JSON file
def save_global_poses(global_poses, filename):
    global_poses_list = [pose.tolist() for pose in global_poses]
    with open(filename, 'w') as f:
        json.dump({"global_poses": global_poses_list}, f)
    print(f"Global poses saved to {filename}")

# Function to construct the global map
def construct_global_map(global_poses, image_paths):
    orb = cv2.ORB_create()
    global_map = []

    for i, image_path in enumerate(image_paths):
        frame = cv2.imread(image_path)
        keypoints, descriptors = orb.detectAndCompute(frame, None)

        # Transform keypoints to global coordinates
        points_3d = []
        for kp in keypoints:
            point = np.array([kp.pt[0], kp.pt[1], 1.0])
            depth = 1.0  # Replace with actual depth value if available
            point_3d = point * depth
            point_3d = np.append(point_3d, 1.0)

            global_point_3d = global_poses[i] @ point_3d
            points_3d.append(global_point_3d[:3])

        global_map.extend(points_3d)

    return np.array(global_map)

# Function to save global map to a file
def save_global_map(global_map, filename):
    np.savetxt(filename, global_map, delimiter=',')
    print(f"Global map saved to {filename}")

# Paths to the results files from local optimization and loop closure detection
local_optimization_results_file = 'local_optimization_results.json'
loop_closure_results_file = 'loop_closure_results.json'

# Load results
poses, image_paths = load_local_optimization_results(local_optimization_results_file)
loop_closures = load_loop_closure_results(loop_closure_results_file)

# Perform global optimization
optimized_poses = global_optimization(poses, loop_closures)

# Compute global poses
global_poses = compute_global_poses(optimized_poses)

# Save global poses to a JSON file
global_poses_file = 'global_poses.json'
save_global_poses(global_poses, global_poses_file)

# Construct the global map
global_map = construct_global_map(global_poses, image_paths)

# Save the global map to a file
save_global_map(global_map, 'global_map.csv')
