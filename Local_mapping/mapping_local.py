import numpy as np
import open3d as o3d
from transformers import pipeline
from PIL import Image
import cv2
from feature_SIFT import load_images_from_folder, extract_features
from SLAM01_1.feature_match import match_features

#initialize depth estimation
depth_pipe = pipeline(task= "depth-estimation", model="depth-anything/Depth-Anything-V2-Small-hf")

#camera intrinsic parameters

fx = 525.0
fy = 525.0
cx = 319.5
cy = 239.5

k = np.array([[fx,0, cx],
              [0, fy, cy],
              [0, 0, 1]])

def create_point_cloud(image , depth, fx , fy , cx , cy):
    """
    create a point cloud from single image nad depth map
    :param image: image as numpy array
    :param depth:
    :param fx:
    :param fy:
    :param cx:
    :param cy:
    :return: Open3D PointCloud object
    """
    height, width = depth.shape
    mask = depth > 0

    u, v = np.meshgrid(np.arange(width), np.arange(height))
    u = u[mask]
    v = v[mask]
    z = depth[mask]

    x = (u - cx) * z/fx
    y = (v -cy) * z/fy

    points = np.vstack((x, y, z)).T

    colors = image[mask]
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    pcd.colors = o3d.utility.Vector3dVector(colors.astype(np.float64)/255.0)

    return pcd

def integrate_local_map(images, depth , fx, fy, cx, cy):
    """
    Integrate local maps into a global point cloud map
    :param images:
    :param depth:
    :param fx:
    :param fy:
    :param cx:
    :param cy:
    :return:  Integarted Open3D Pointcloud objects
    """

    global_map = o3d.geometry.PointCloud()

    for images, depth in zip(images, depth_maps):
        local_map = create_point_cloud(image, depth, fx , fy , cx , cy)
        global_map += local_map

    global_map = global_map.voxel_down_sample(voxel_size=0.02)
    return global_map

#usage

# Example usage:
if __name__ == "__main__":
    # Load images
    images = load_images_from_folder('/home/thales1/VSLAM/Vslam_algorithm/pythonProject1/Visual_slam_optimized/DATA/Data2')

    # Extract features from loaded images
    features = extract_features(images, visualize=False)

    # Perform feature matching on the extracted features
    matches = match_features(features)

    K_inv = np.linalg.inv(k)

    depth_maps = []
    global_map = o3d.geometry.PointCloud()

    for i, (img1_name, img2_name, good_matches) in enumerate(matches):
        kp1 = [kp for (name, kp, des) in features if name == img1_name][0]
        kp2 = [kp for (name, kp, des) in features if name == img2_name][0]

        # Apply RANSAC to reject outliers
        good_matches_inliers, inlier_mask, _ = reject_outliers_ransac(kp1, kp2, good_matches)

        if not good_matches_inliers:
            print(f"No good matches found between {img1_name} and {img2_name} after RANSAC.")
            continue

        # Load the corresponding images
        img1 = next(img for (name, img) in images if name == img1_name)
        img2 = next(img for (name, img) in images if name == img2_name)

        # Convert image to PIL format for depth estimation
        img1_pil = Image.fromarray(cv2.cvtColor(img1, cv2.COLOR_BGR2RGB))
        img2_pil = Image.fromarray(cv2.cvtColor(img2, cv2.COLOR_BGR2RGB))

        # Perform depth estimation
        depth_map1 = np.array(depth_pipe(img1_pil)["depth"])
        depth_map2 = np.array(depth_pipe(img2_pil)["depth"])

        depth_maps.append(depth_map1)

        # Calculate optical flow using the good inlier matches
        pts1_flow, pts2_flow, status = calculate_optical_flow(img1, img2, kp1, kp2, good_matches_inliers)

        if pts1_flow is None or pts2_flow is None:
            print(f"Optical flow calculation failed for {img1_name} and {img2_name}.")
            continue

        # Convert the 2D points from the first image to 3D using depth map
        pts1_3D = pixel_to_camera_coords(pts1_flow, depth_map1, K_inv)

        # Ensure that the number of 3D points matches the number of 2D points
        if pts1_3D.shape[0] != pts2_flow.shape[0]:
            print(f"Mismatch in the number of 3D and 2D points: {pts1_3D.shape[0]} 3D points, {pts2_flow.shape[0]} 2D points")
            continue

        # Estimate relative pose between the frames
        rvec, tvec, inliers = estimate_relative_pose(pts1_3D, pts2_flow, K)

        if rvec is not None and tvec is not None:
            print(f"Relative pose between {img1_name} and {img2_name}:")
            print(f"Rotation Vector (rvec): {rvec}")
            print(f"Translation Vector (tvec): {tvec}")

            # Create and integrate the local map for the current frame
            local_map = create_point_cloud(img1, depth_map1, fx, fy, cx, cy)
            global_map += local_map

        else:
            print(f"Pose estimation failed for {img1_name} and {img2_name}.")

    # Downsample the global map for visualization and performance
    global_map = global_map.voxel_down_sample(voxel_size=0.02)
    o3d.visualization.draw_geometries([global_map], window_name="Global Point Cloud Map")
