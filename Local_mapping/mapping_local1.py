import numpy as np
import open3d as o3d
from typing import List
from feature_SIFT import load_images_from_folder, extract_features
from estimating_depth import estimate_depth_maps

# Camera intrinsic parameters
fx, fy, cx, cy = 525.0, 525.0, 319.5, 239.5
K = np.array([[fx, 0, cx],
              [0, fy, cy],
              [0, 0, 1]])

def create_local_map(image: np.ndarray, depth_map: np.ndarray, feature_points: np.ndarray) -> o3d.geometry.PointCloud:
    """
    Create a local point cloud map from an image, its depth map, and feature points.

    Args:
    - image: RGB image as a numpy array
    - depth_map: Depth map as a numpy array
    - feature_points: 2D feature points as a numpy array of shape (N, 2)

    Returns:
    - local_map: Open3D PointCloud object
    """
    height, width = depth_map.shape

    # Extract 3D points for feature points
    u, v = feature_points[:, 0], feature_points[:, 1]
    z = depth_map[np.round(v).astype(int), np.round(u).astype(int)]
    x = (u - cx) * z / fx
    y = (v - cy) * z / fy

    points = np.vstack((x, y, z)).T

    # Extract colors for feature points
    colors = image[np.round(v).astype(int), np.round(u).astype(int)]

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    pcd.colors = o3d.utility.Vector3dVector(colors.astype(np.float64) / 255.0)

    return pcd

def local_mapping(image_folder: str) -> List[o3d.geometry.PointCloud]:
    """
    Perform local mapping using images from a folder.

    Args:
    - image_folder: Path to the folder containing image sequence

    Returns:
    - local_maps: List of Open3D PointCloud objects representing local maps
    """
    # Load images and extract features
    images = load_images_from_folder(image_folder)
    features = extract_features(images, visualize=False)

    # Estimate depth maps
    depth_maps = estimate_depth_maps(images, visualize=False)

    # Create local maps
    local_maps = []
    for (image_name, image), (_, keypoints, _), depth_map in zip(images, features, depth_maps):
        feature_points = np.array([kp.pt for kp in keypoints])
        local_map = create_local_map(image, depth_map, feature_points)
        local_maps.append(local_map)

    return local_maps

def visualize_local_maps(local_maps: List[o3d.geometry.PointCloud]):
    """
    Visualize the local point cloud maps.

    Args:
    - local_maps: List of Open3D PointCloud objects
    """
    # Combine all local maps into a single point cloud for visualization
    combined_map = o3d.geometry.PointCloud()
    for i, local_map in enumerate(local_maps):
        # Offset each local map slightly for better visualization
        offset = np.array([i * 0.5, 0, 0])  # Offset in x-direction
        local_map.translate(offset)
        combined_map += local_map

    # Downsample the combined map for better visualization performance
    combined_map = combined_map.voxel_down_sample(voxel_size=0.02)

    o3d.visualization.draw_geometries([combined_map], window_name="Local Point Cloud Maps")

if __name__ == "__main__":
    image_folder = '/home/thales1/VSLAM/Vslam_algorithm/pythonProject1/Visual_slam_optimized/DATA/Data2'
    local_maps = local_mapping(image_folder)
    print(f"Created {len(local_maps)} local maps")
    visualize_local_maps(local_maps)