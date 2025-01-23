import numpy as np
import cv2
from feature_SIFT import load_images_from_folder, extract_features
from SLAM01_1.feature_match import match_features
from Reject_outlier import reject_outliers_ransac
from Opticflow_estimation import optical_flow
from relative_pose_estimation import estimate_relative_pose, pixel_to_camera_coords
from estimating_depth import estimate_depth_maps


def visual_odometry_estimation(image_folder):
    # Load images
    images = load_images_from_folder(image_folder)

    # Extract features
    features = extract_features(images, visualize=False)

    # Estimate depth maps
    depth_maps = estimate_depth_maps(images, visualize=False)

    # Perform feature matching
    matches = match_features(features)

    # Camera intrinsic matrix (from relative_pose_estimation.py)
    K = np.array([[525.0, 0, 319.5],
                  [0, 525.0, 239.5],
                  [0, 0, 1]])
    K_inv = np.linalg.inv(K)

    trajectory = []
    current_pose = np.eye(4)  # Start with identity matrix (no movement)

    for i, (img1_name, img2_name, good_matches) in enumerate(matches):
        kp1 = [kp for (name, kp, des) in features if name == img1_name][0]
        kp2 = [kp for (name, kp, des) in features if name == img2_name][0]

        # Outlier rejection
        good_matches_inliers, inlier_mask, _ = reject_outliers_ransac(kp1, kp2, good_matches)

        if not good_matches_inliers:
            print(f"No good matches found between {img1_name} and {img2_name} after RANSAC.")
            continue

        # Load the corresponding images and depth maps
        img1 = next(img for (name, img) in images if name == img1_name)
        img2 = next(img for (name, img) in images if name == img2_name)
        depth_map1 = depth_maps[i]
        depth_map2 = depth_maps[i + 1]

        # Optical flow estimation
        pts1_flow, pts2_flow, status = optical_flow(img1, img2, kp1, kp2, good_matches_inliers)

        if pts1_flow is None or pts2_flow is None:
            print(f"Optical flow calculation failed for {img1_name} and {img2_name}.")
            continue

        # Convert 2D points to 3D using depth information
        pts1_3D = pixel_to_camera_coords(pts1_flow, depth_map1, K_inv)

        # Relative pose estimation
        rvec, tvec, inliers = estimate_relative_pose(pts1_3D, pts2_flow, K)

        if rvec is not None and tvec is not None:
            # Convert rotation vector to rotation matrix
            R, _ = cv2.Rodrigues(rvec)

            # Construct 4x4 transformation matrix
            T = np.eye(4)
            T[:3, :3] = R
            T[:3, 3] = tvec.reshape(3)

            # Update current pose
            current_pose = current_pose @ np.linalg.inv(T)

            # Add current position to trajectory
            trajectory.append(current_pose[:3, 3])

            print(f"Estimated pose between {img1_name} and {img2_name}:")
            print(current_pose)
        else:
            print(f"Pose estimation failed for {img1_name} and {img2_name}.")

    return np.array(trajectory)


# Example usage
if __name__ == "__main__":
    image_folder = '/home/thales1/VSLAM/Vslam_algorithm/pythonProject1/Visual_slam_optimized/DATA/Data2'
    trajectory = visual_odometry_estimation(image_folder)

    # Plot trajectory
    import matplotlib.pyplot as plt

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.plot(trajectory[:, 0], trajectory[:, 1], trajectory[:, 2])
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title('Estimated Camera Trajectory')
    plt.show()