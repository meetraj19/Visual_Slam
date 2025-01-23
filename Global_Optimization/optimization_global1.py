import numpy as np
from scipy.optimize import least_squares
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import logging
from typing import List, Tuple, Optional
import cv2

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def pose_error(params, n_poses, edges, measurements):
    """Compute the error for pose graph optimization."""
    poses = params.reshape((n_poses, 6))
    error = []
    for (i, j), measurement in zip(edges, measurements):
        pose_i = poses[i]
        pose_j = poses[j]
        T_i = pose_to_transform(pose_i)
        T_j = pose_to_transform(pose_j)
        T_ij_measured = pose_to_transform(measurement)
        T_ij_estimated = np.linalg.inv(T_i) @ T_j
        error_ij = transform_to_pose(T_ij_measured @ np.linalg.inv(T_ij_estimated))
        error.append(error_ij)
    return np.concatenate(error)


def pose_to_transform(pose):
    """Convert a 6D pose vector to a 4x4 transformation matrix."""
    T = np.eye(4)
    T[:3, :3] = euler_to_rotation_matrix(pose[:3])
    T[:3, 3] = pose[3:]
    return T


def transform_to_pose(T):
    """Convert a 4x4 transformation matrix to a 6D pose vector."""
    pose = np.zeros(6)
    pose[:3] = rotation_matrix_to_euler(T[:3, :3])
    pose[3:] = T[:3, 3]
    return pose


def euler_to_rotation_matrix(euler):
    """Convert Euler angles to rotation matrix."""
    Rx = np.array([[1, 0, 0],
                   [0, np.cos(euler[0]), -np.sin(euler[0])],
                   [0, np.sin(euler[0]), np.cos(euler[0])]])
    Ry = np.array([[np.cos(euler[1]), 0, np.sin(euler[1])],
                   [0, 1, 0],
                   [-np.sin(euler[1]), 0, np.cos(euler[1])]])
    Rz = np.array([[np.cos(euler[2]), -np.sin(euler[2]), 0],
                   [np.sin(euler[2]), np.cos(euler[2]), 0],
                   [0, 0, 1]])
    return Rz @ Ry @ Rx


def rotation_matrix_to_euler(R):
    """Convert rotation matrix to Euler angles."""
    sy = np.sqrt(R[0, 0] * R[0, 0] + R[1, 0] * R[1, 0])
    singular = sy < 1e-6
    if not singular:
        x = np.arctan2(R[2, 1], R[2, 2])
        y = np.arctan2(-R[2, 0], sy)
        z = np.arctan2(R[1, 0], R[0, 0])
    else:
        x = np.arctan2(-R[1, 2], R[1, 1])
        y = np.arctan2(-R[2, 0], sy)
        z = 0
    return np.array([x, y, z])


def optimize_pose_graph(poses, loop_closures):
    """Optimize the pose graph using scipy's least_squares."""
    n_poses = len(poses)
    initial_poses = np.array([transform_to_pose(pose) for pose in poses])

    edges = [(i, i + 1) for i in range(n_poses - 1)] + loop_closures
    measurements = [transform_to_pose(np.linalg.inv(poses[i]) @ poses[j]) for i, j in edges]

    result = least_squares(pose_error, initial_poses.ravel(),
                           args=(n_poses, edges, measurements),
                           verbose=2)

    optimized_poses = [pose_to_transform(pose) for pose in result.x.reshape((n_poses, 6))]
    return optimized_poses


def visualize_trajectory(poses: List[np.ndarray], optimized_poses: Optional[List[np.ndarray]] = None,
                         loop_closures: Optional[List[Tuple[int, int]]] = None):
    """Visualize the camera trajectory and loop closures."""
    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection='3d')

    # Plot original trajectory
    x, y, z = zip(*[pose[:3, 3] for pose in poses])
    ax.plot(x, y, z, 'r-', label='Original Trajectory')

    # Plot optimized trajectory if available
    if optimized_poses is not None:
        x, y, z = zip(*[pose[:3, 3] for pose in optimized_poses])
        ax.plot(x, y, z, 'g-', label='Optimized Trajectory')

    # Plot loop closures if available
    if loop_closures is not None:
        for i, j in loop_closures:
            if isinstance(i, int) and isinstance(j, int):
                p1 = poses[i][:3, 3]
                p2 = poses[j][:3, 3]
                ax.plot([p1[0], p2[0]], [p1[1], p2[1]], [p1[2], p2[2]], 'b--', alpha=0.5)

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.legend()
    plt.title('Camera Trajectory')
    plt.show()

def fallback_loop_closure_detection(features, threshold=0.75):
    """A simple fallback method for loop closure detection."""
    loop_closures = []
    n = len(features)
    for i in range(n):
        for j in range(i + 10, n):  # Skip nearby frames
            desc1 = features[i][2]
            desc2 = features[j][2]
            # Use L2 norm to compare descriptors
            matches = cv2.BFMatcher(cv2.NORM_L2, crossCheck=True).match(desc1, desc2)
            similarity = len(matches) / max(len(desc1), len(desc2))
            if similarity > threshold:
                loop_closures.append((i, j))
    return loop_closures


def global_optimization(local_poses: List[np.ndarray], features: List[Tuple[str, List, np.ndarray]]) -> List[
    np.ndarray]:
    """
    Perform global optimization using pose graph optimization.

    Args:
    - local_poses: List of camera poses from local optimization
    - features: List of tuples (image_name, keypoints, descriptors) for loop closure detection

    Returns:
    - List of globally optimized camera poses
    """
    try:
        # Perform loop closure detection
        loop_closures = fallback_loop_closure_detection(features)

        logger.info(f"Detected {len(loop_closures)} loop closures")

        if not loop_closures:
            logger.warning("No loop closures detected. Optimization may not improve results significantly.")

        # Optimize pose graph
        optimized_poses = optimize_pose_graph(local_poses, loop_closures)

        # Visualize results
        visualize_trajectory(local_poses, optimized_poses, loop_closures)

        return optimized_poses
    except Exception as e:
        logger.error(f"Error during global optimization: {str(e)}")
        return local_poses


def compute_trajectory_error(original_poses: List[np.ndarray], optimized_poses: List[np.ndarray]) -> float:
    """Compute the average error between original and optimized trajectories."""
    errors = []
    for orig, opt in zip(original_poses, optimized_poses):
        error = np.linalg.norm(orig[:3, 3] - opt[:3, 3])
        errors.append(error)
    return np.mean(errors)


if __name__ == "__main__":
    # Simulated input (replace with actual data from local optimization)
    n_frames = 100
    local_poses = [np.eye(4) for _ in range(n_frames)]
    for i in range(1, n_frames):
        local_poses[i][:3, 3] = np.random.randn(3) * 0.1 + local_poses[i - 1][:3, 3]

    # Simulated features (replace with actual features)
    features = [(f"frame_{i}", [], np.random.rand(100, 128).astype(np.float32)) for i in range(n_frames)]

    # Perform global optimization
    optimized_poses = global_optimization(local_poses, features)

    # Compute and log trajectory error
    error = compute_trajectory_error(local_poses, optimized_poses)
    logger.info(f"Average trajectory error: {error:.4f}")

    logger.info("Global optimization completed.")
    logger.info(f"Original trajectory start: {local_poses[0][:3, 3]}")
    logger.info(f"Original trajectory end: {local_poses[-1][:3, 3]}")
    logger.info(f"Optimized trajectory start: {optimized_poses[0][:3, 3]}")
    logger.info(f"Optimized trajectory end: {optimized_poses[-1][:3, 3]}")