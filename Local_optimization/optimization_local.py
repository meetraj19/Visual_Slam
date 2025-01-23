import numpy as np
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation as R
from scipy.sparse import lil_matrix
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def Project(points, camera_matrix, ext_matrix):
    """
    Project 3D points into 2D using the camera matrix and extrinsic matrix
    :param points: 3D points in world coordinate
    :param camera_matrix: intrinsic matrix
    :param ext_matrix:  extrinsic matrix
    :return: ndarray 2D projected points
    """

    points_h = np.hstack([points, np.ones((points.shape[0], 1))])
    P = camera_matrix @ ext_matrix
    projected_points = (P @ points_h.T).T
    return projected_points[:,:2] / projected_points[:,:2]


def reprojection_error(params, n_cameras, n_points, camera_indices, point_indices, points_2d):
    """
    compute reprojection error for all camera-point correspondences
    :param params: flattened camera parameters and 3D points
    :param n_cameras:
    :param n_points:
    :param camera_indices:
    :param point_indices:
    :param points_2d:
    :return: Residual errors between observed and projected points
    """

    camera_params = params[:n_cameras*9].reshape((n_cameras, 9))
    points_3d = params[n_cameras*9:].reshape((n_points,3))

    points_proj = np.zeros_like(points_2d)

    for i, (cam_idx, point_idx) in enumerate(zip(camera_indices, point_indices)):
        camera = camera_params(cam_idx)
        point = points_3d(point_idx)

        r_vec = camera[:3]
        t_vec = camera[3:6]
        f , cx, cy = camera[6:]
        camera_matrix = np.array([[f, 0, cx],
                                  [0, f, cy],
                                  [0, 0, 1]])
        r_mat = R.from_rotvec(r_vec).as_matrix()
        ext_matrix = np.hstack((r_mat, t_vec.reshape(3,1)))

        points_proj[i] = project(point.reshape(1,-1), camera_matrix, ext_matrix).flatten()

    return (points_proj- points_2d).ravel()

def bundle_adjustment(camera_params, points_3d, camera_indices, point_indices, points_2d):
    """
    Perform bundle adjustment to optimize camera parameters and 3D points.

    Args:
    - camera_params (ndarray): Initial camera parameters.
    - points_3d (ndarray): Initial 3D points.
    - camera_indices (ndarray): Indices mapping 2D points to cameras.
    - point_indices (ndarray): Indices mapping 2D points to 3D points.
    - points_2d (ndarray): Observed 2D points.

    Returns:
    - optimized_camera_params (ndarray): Optimized camera parameters.
    - optimized_points_3d (ndarray): Optimized 3D points.
    """
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
                            args=(n_cameras, n_points, camera_indices, point_indices, points_2d))

        optimized_params = res.x
        optimized_camera_params = optimized_params[:n_cameras * 9].reshape((n_cameras, 9))
        optimized_points_3d = optimized_params[n_cameras * 9:].reshape((n_points, 3))

        logger.info(f"Bundle adjustment completed. Initial cost: {res.cost}, Final cost: {res.fun}")
        return optimized_camera_params, optimized_points_3d
    except Exception as e:
        logger.error(f"Error in bundle adjustment: {str(e)}")
        return camera_params, points_3d

def check_input_validity(camera_params, points_3d, camera_indices, point_indices, points_2d):
    """
    Validate the inputs for bundle adjustment.

    Args:
    - camera_params (ndarray): Camera parameters.
    - points_3d (ndarray): 3D points.
    - camera_indices (ndarray): Indices mapping 2D points to cameras.
    - point_indices (ndarray): Indices mapping 2D points to 3D points.
    - points_2d (ndarray): Observed 2D points.

    Raises:
    - AssertionError: If any input is invalid.
    """
    assert camera_params.shape[1] == 9, "Camera parameters should have 9 elements per camera"
    assert points_3d.shape[1] == 3, "3D points should have 3 coordinates"
    assert len(camera_indices) == len(point_indices) == len(
        points_2d), "Indices and 2D points should have the same length"
    assert np.max(camera_indices) < len(camera_params), "Camera index out of bounds"
    assert np.max(point_indices) < len(points_3d), "Point index out of bounds"

# Example usage:
if __name__ == "__main__":
    # Initialize example data (you will replace this with real data from your SLAM pipeline)
    n_cameras = 5
    n_points = 100
    n_observations = 200

    camera_params = np.random.randn(n_cameras, 9)
    points_3d = np.random.randn(n_points, 3)
    camera_indices = np.random.randint(0, n_cameras, n_observations)
    point_indices = np.random.randint(0, n_points, n_observations)
    points_2d = np.random.randn(n_observations, 2)

    # Check input validity
    check_input_validity(camera_params, points_3d, camera_indices, point_indices, points_2d)

    # Perform bundle adjustment
    optimized_camera_params, optimized_points_3d = bundle_adjustment(camera_params, points_3d, camera_indices, point_indices, points_2d)

    logger.info(f"Optimized camera parameters: {optimized_camera_params}")
    logger.info(f"Optimized 3D points: {optimized_points_3d}")