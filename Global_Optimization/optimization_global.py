import cv2
import numpy as np
from scipy.optimize import least_squares
import logging

# Setup logging for debugging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def pose_error(params, n_poses, edges, measurements, information_matrices):
    """
    Compute the error for pose graph optimization.

    Args:
    - params (numpy.ndarray): Optimizable parameters (camera poses).
    - n_poses (int): Number of poses.
    - edges (list of tuple): List of pose pairs (i, j) with an edge.
    - measurements (list of numpy.ndarray): Relative pose measurements.
    - information_matrices (list of numpy.ndarray): Information matrices for each measurement.

    Returns:
    - error (numpy.ndarray): Error for each edge.
    """
    poses = params.reshape((n_poses, 6))

    error = []
    for (i, j), measurement, info_matrix in zip(edges, measurements, information_matrices):
        pose_i = poses[i]
        pose_j = poses[j]

        # Convert the pose parameters to transformation matrices
        transform_i = pose_to_transform(pose_i)
        transform_j = pose_to_transform(pose_j)
        transform_measurement = pose_to_transform(measurement)

        # Compute the relative transformation error
        relative_transform = np.linalg.inv(transform_i) @ transform_j
        transform_error = transform_measurement @ np.linalg.inv(relative_transform)

        # Convert the transformation error to a 6D error
        error_6d = transform_to_pose(transform_error)
        weighted_error = info_matrix @ error_6d

        error.append(weighted_error)

    return np.concatenate(error)

def pose_to_transform(pose):
    """
    Convert a 6D pose to a 4x4 transformation matrix.

    Args:
    - pose (numpy.ndarray): 6D pose (3 for rotation and 3 for translation).

    Returns:
    - transform (numpy.ndarray): 4x4 transformation matrix.
    """
    rvec, tvec = pose[:3], pose[3:]
    rot_matrix, _ = cv2.Rodrigues(rvec)
    transform = np.eye(4)
    transform[:3, :3] = rot_matrix
    transform[:3, 3] = tvec
    return transform

def transform_to_pose(transform):
    """
    Convert a 4x4 transformation matrix to a 6D pose.

    Args:
    - transform (numpy.ndarray): 4x4 transformation matrix.

    Returns:
    - pose (numpy.ndarray): 6D pose (3 for rotation and 3 for translation).
    """
    rvec, _ = cv2.Rodrigues(transform[:3, :3])
    tvec = transform[:3, 3]
    return np.hstack([rvec.flatten(), tvec])

def optimize_pose_graph(initial_poses, edges, measurements, information_matrices):
    """
    Optimize the pose graph using least squares optimization.

    Args:
    - initial_poses (numpy.ndarray): Initial poses (n_poses, 6).
    - edges (list of tuple): List of pose pairs (i, j) with an edge.
    - measurements (list of numpy.ndarray): Relative pose measurements.
    - information_matrices (list of numpy.ndarray): Information matrices for each measurement.

    Returns:
    - optimized_poses (numpy.ndarray): Optimized poses (n_poses, 6).
    """
    n_poses = initial_poses.shape[0]
    x0 = initial_poses.ravel()

    logger.info("Starting pose graph optimization...")

    res = least_squares(pose_error, x0, args=(n_poses, edges, measurements, information_matrices), verbose=2)

    optimized_poses = res.x.reshape((n_poses, 6))
    logger.info("Pose graph optimization completed.")
    return optimized_poses

def check_input_validity(initial_poses, edges, measurements, information_matrices):
    """
    Validate the inputs for pose graph optimization.

    Args:
    - initial_poses (numpy.ndarray): Initial poses (n_poses, 6).
    - edges (list of tuple): List of pose pairs (i, j) with an edge.
    - measurements (list of numpy.ndarray): Relative pose measurements.
    - information_matrices (list of numpy.ndarray): Information matrices for each measurement.

    Raises:
    - ValueError: If any input is invalid.
    """
    if initial_poses.shape[1] != 6:
        raise ValueError("Initial poses should have 6 elements per pose.")
    if len(edges) != len(measurements) or len(edges) != len(information_matrices):
        raise ValueError("Edges, measurements, and information matrices should have the same length.")
    for i, ((_, _), meas, info) in enumerate(zip(edges, measurements, information_matrices)):
        if meas.shape != (6,):
            raise ValueError(f"Measurement {i} should have 6 elements.")
        if info.shape != (6, 6):
            raise ValueError(f"Information matrix {i} should be 6x6.")

# Example usage:
if __name__ == "__main__":
    # Initialize example data (replace these with actual data from your SLAM pipeline)
    n_poses = 3
    initial_poses = np.random.randn(n_poses, 6)  # Random initialization, replace with actual initial poses
    edges = [(0, 1), (1, 2), (2, 0)]  # Example edges, replace with actual edges
    measurements = [np.random.randn(6) for _ in edges]  # Example measurements, replace with actual measurements
    information_matrices = [np.eye(6) for _ in edges]  # Example information matrices, replace with actual matrices

    # Validate input data
    check_input_validity(initial_poses, edges, measurements, information_matrices)

    # Optimize the pose graph
    optimized_poses = optimize_pose_graph(initial_poses, edges, measurements, information_matrices)

    # Output the optimized poses
    logger.info(f"Optimized poses:\n{optimized_poses}")
