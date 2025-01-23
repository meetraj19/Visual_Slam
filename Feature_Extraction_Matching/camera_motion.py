import cv2
import numpy as np
import glob
import os
import json


def estimate_camera_motion(image_dir, camera_matrix, dist_coeffs):
    # Initialize ORB detector
    orb = cv2.ORB_create()

    # Initialize Matcher
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

    # Initialize camera poses with identity matrix
    poses = [np.eye(4)]

    # Get a sorted list of .png image files
    image_paths = sorted(glob.glob(os.path.join(image_dir, '*.png')))

    # Check if images are found
    if not image_paths:
        raise ValueError("No .png images found in the directory.")

    # Placeholder for previous descriptors and keypoints
    prev_descriptors = None
    prev_keypoints = None

    # Loop through all image paths
    for idx, image_path in enumerate(image_paths):
        try:
            # Load image
            current_image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
            if current_image is None:
                raise ValueError(f"Image {image_path} could not be loaded.")

            # Detect ORB features and compute descriptors
            current_keypoints, current_descriptors = orb.detectAndCompute(current_image, None)
            if current_keypoints is None or current_descriptors is None:
                print(f"No keypoints or descriptors found in image {idx}.")
                continue

            # If not the first image, match descriptors and estimate motion
            if prev_descriptors is not None:
                # Match descriptors with the previous image
                matches = bf.match(prev_descriptors, current_descriptors)
                matches = sorted(matches, key=lambda x: x.distance)

                # Extract locations of matched keypoints in both images
                pts1 = np.float32([prev_keypoints[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
                pts2 = np.float32([current_keypoints[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)

                # Find the Essential matrix
                E, mask = cv2.findEssentialMat(pts1, pts2, camera_matrix, cv2.FM_RANSAC, 0.999, 1.0)
                if E is None or mask is None:
                    print(f"Essential matrix could not be computed for image {idx}.")
                    continue

                # Recover the relative camera rotation and translation from the Essential matrix
                _, R, t, mask = cv2.recoverPose(E, pts1, pts2, camera_matrix)
                if R is None or t is None:
                    print(f"Pose could not be recovered for image {idx}.")
                    continue

                # Construct the transformation matrix
                pose = np.eye(4)
                pose[:3, :3] = R
                pose[:3, 3] = t.ravel()

                # Update the global pose
                poses.append(poses[-1] @ np.linalg.inv(pose))

            # Update previous image keypoints and descriptors
            prev_keypoints = current_keypoints
            prev_descriptors = current_descriptors

        except Exception as e:
            print(f"An error occurred while processing image {idx}: {e}")

    return poses


def save_poses_to_json(poses, filename="camera_poses.json"):
    poses_data = [{"pose_index": i, "pose_matrix": pose.tolist()} for i, pose in enumerate(poses)]

    with open(filename, 'w') as json_file:
        json.dump(poses_data, json_file, indent=4)


# Directory where images are stored
image_dir = '/home/thales1/VSLAM/a/pythonProject1/DATA/Data2'

# Camera intrinsic parameters (assumed values, replace with calibration values)
focal_length = 800  # focal length
image_center_x = 640  # principal point x (based on a 1280x720 image)
image_center_y = 360  # principal point y (based on a 1280x720 image)
camera_matrix = np.array([[focal_length, 0, image_center_x],
                          [0, focal_length, image_center_y],
                          [0, 0, 1]])
dist_coeffs = np.zeros(5)  # Assuming no lens distortion

# Estimate the camera motion
camera_poses = estimate_camera_motion(image_dir, camera_matrix, dist_coeffs)

# Save the camera poses to a JSON file
save_poses_to_json(camera_poses)
