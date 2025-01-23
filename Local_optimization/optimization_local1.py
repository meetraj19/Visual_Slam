
from Estimating_VO import visual_odometry_estimation
from mapping_local1 import local_mapping

import numpy as np
import cv2
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation as R
from scipy.sparse import lil_matrix
import logging
from typing import List, Tuple


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Camera intrinsic parameters
fx, fy, cx, cy = 525.0, 525.0, 319.5, 239.5
K = np.array([[fx, 0, cx],
              [0, fy, cy],
              [0, 0, 1]])


def project(points, camera_matrix, ext_matrix):
    points_h = np.hstack([points, np.ones((points.shape[0], 1))])
    P = camera_matrix @ ext_matrix
    projected_points = (P @ points_h.T).T

    # Avoid division by zero by adding a small epsilon value
    z_coords = projected_points[:, 2:]
    z_coords[z_coords == 0] = 1e-6

    return projected_points[:, :2] / z_coords


def reprojection_error(params: np.ndarray, n_cameras: int, n_points: int, camera_indices: np.ndarray,
                       point_indices: np.ndarray, points_2d: np.ndarray, K: np.ndarray) -> np.ndarray:
    """Compute reprojection error for all camera-point correspondences."""
    camera_params = params[:n_cameras * 6].reshape((n_cameras, 6))
    points_3d = params[n_cameras * 6:].reshape((n_points, 3))

    points_proj = np.zeros_like(points_2d)

    for i, (cam_idx, point_idx) in enumerate(zip(camera_indices, point_indices)):
        camera = camera_params[cam_idx]
        point = points_3d[point_idx]

        r_vec = camera[:3]
        t_vec = camera[3:6]
        r_mat = R.from_rotvec(r_vec).as_matrix()
        ext_matrix = np.hstack((r_mat, t_vec.reshape(3, 1)))

        points_proj[i] = project(point.reshape(1, -1), K, ext_matrix).flatten()

    return (points_proj - points_2d).ravel()


def bundle_adjustment(camera_params, points_3d, camera_indices, point_indices, points_2d, K):
    # Ensure no NaNs or infinities in camera parameters and points
    if np.any(np.isnan(camera_params)) or np.any(np.isnan(points_3d)):
        raise ValueError("Initial camera parameters or 3D points contain NaN values.")
    if np.any(np.isinf(camera_params)) or np.any(np.isinf(points_3d)):
        raise ValueError("Initial camera parameters or 3D points contain infinite values.")

    try:
        n_cameras = camera_params.shape[0]
        n_points = points_3d.shape[0]
        n_observations = len(camera_indices)

        x0 = np.hstack((camera_params.ravel(), points_3d.ravel()))

        A = lil_matrix((2 * n_observations, 9 * n_cameras + 3 * n_points), dtype=int)

        i = np.arange(n_observations)
        for s in range(9):
            A[2 * i, camera_indices * 9 + s] = 1
            A[2 * i + 1, camera_indices * 9 + s] = 1
        for s in range(3):
            A[2 * i, n_cameras * 9 + point_indices * 3 + s] = 1
            A[2 * i + 1, n_cameras * 9 + point_indices * 3 + s] = 1

        A = A.tocsr()

        res = least_squares(reprojection_error, x0, jac_sparsity=A, verbose=2, x_scale='jac', ftol=1e-4, method='trf',
                            args=(n_cameras, n_points, camera_indices, point_indices, points_2d, K))

        optimized_params = res.x
        optimized_camera_params = optimized_params[:n_cameras * 6].reshape((n_cameras, 6))
        optimized_points_3d = optimized_params[n_cameras * 6:].reshape((n_points, 3))

        return optimized_camera_params, optimized_points_3d
    except Exception as e:
        print(f"Error during bundle adjustment: {e}")
        return camera_params, points_3d


def prepare_optimization_data(relative_poses: List[np.ndarray], local_maps: List[np.ndarray],
                              feature_matches: List[List[cv2.DMatch]]) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Prepare data for optimization from relative pose estimation and local mapping results."""
    camera_params = []
    points_3d = []
    camera_indices = []
    point_indices = []
    points_2d = []

    for i, pose in enumerate(relative_poses):
        r_vec, _ = cv2.Rodrigues(pose[:3, :3])
        t_vec = pose[:3, 3]
        camera_params.append(np.hstack((r_vec.flatten(), t_vec)))

        if i < len(feature_matches):
            for j, match in enumerate(feature_matches[i]):
                if match.trainIdx < len(local_maps[i].points) and match.queryIdx < len(local_maps[i+1].points):
                    points_3d.append(local_maps[i].points[match.trainIdx])
                    camera_indices.append(i)
                    point_indices.append(len(points_3d) - 1)
                    points_2d.append(local_maps[i+1].points[match.queryIdx][:2])  # Only take x, y coordinates

    return (np.array(camera_params), np.array(points_3d),
            np.array(camera_indices), np.array(point_indices), np.array(points_2d))


def local_optimization(image_folder: str) -> Tuple[List[np.ndarray], np.ndarray]:
    """Perform local optimization using results from relative pose estimation and local mapping."""
    # Get visual odometry results
    odometry_results = visual_odometry_estimation(image_folder)
    relative_poses = [np.eye(4)]  # Start with identity matrix for the first pose
    for rvec, tvec in odometry_results:
        R, _ = cv2.Rodrigues(rvec)
        T = np.eye(4)
        T[:3, :3] = R
        T[:3, 3] = tvec.flatten()
        relative_poses.append(T)

    # Get local mapping results
    local_maps = local_mapping(image_folder)

    # Simulate feature matches (replace this with actual feature matching if available)
    feature_matches = []
    for i in range(len(local_maps) - 1):
        matches = [cv2.DMatch(j, j, 0) for j in range(min(len(local_maps[i].points), len(local_maps[i+1].points)))]
        feature_matches.append(matches)

    logger.info(f"Number of relative poses: {len(relative_poses)}")
    logger.info(f"Number of local maps: {len(local_maps)}")
    logger.info(f"Number of feature match sets: {len(feature_matches)}")

    camera_params, points_3d, camera_indices, point_indices, points_2d = prepare_optimization_data(
        relative_poses, local_maps, feature_matches)

    if len(camera_params) == 0 or len(points_3d) == 0:
        logger.error("No valid data for optimization. Check your inputs.")
        return relative_poses, np.array([])

    optimized_camera_params, optimized_points_3d = bundle_adjustment(
        camera_params, points_3d, camera_indices, point_indices, points_2d, K)

    # Convert optimized camera parameters back to transformation matrices
    optimized_poses = []
    for params in optimized_camera_params:
        r_mat, _ = cv2.Rodrigues(params[:3])
        t_vec = params[3:6]
        pose = np.eye(4)
        pose[:3, :3] = r_mat
        pose[:3, 3] = t_vec
        optimized_poses.append(pose)

    return optimized_poses, optimized_points_3d


if __name__ == "__main__":
    image_folder = '/home/thales1/VSLAM/Vslam_algorithm/pythonProject1/Visual_slam_optimized/DATA/Data2'
    optimized_poses, optimized_points = local_optimization(image_folder)

    logger.info(f"Number of optimized poses: {len(optimized_poses)}")
    logger.info(f"Number of optimized 3D points: {optimized_points.shape[0]}")
