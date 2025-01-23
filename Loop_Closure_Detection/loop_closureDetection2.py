import cv2
import numpy as np
import os
from glob import glob
import json

# Parameters
MIN_MATCH_COUNT = 10
N_CLUSTERS = 500

# Function to extract features and compute descriptors
def extract_features(image, detector):
    keypoints, descriptors = detector.detectAndCompute(image, None)
    return keypoints, descriptors

# Function to perform loop closure detection
def loop_closure_detection(features_db, current_descriptors, threshold=0.75):
    bf = cv2.BFMatcher(cv2.NORM_HAMMING)
    matches = []
    for idx, db_descriptors in enumerate(features_db):
        if db_descriptors is not None:
            raw_matches = bf.knnMatch(current_descriptors, db_descriptors, k=2)
            good_matches = []
            for m, n in raw_matches:
                if m.distance < threshold * n.distance:
                    good_matches.append(m)
            matches.append((idx, len(good_matches)))
    matches = sorted(matches, key=lambda x: x[1], reverse=True)
    return matches[0][0] if matches and matches[0][1] > MIN_MATCH_COUNT else None

# Function to estimate relative pose using feature extraction
def estimate_camera_motion(frame1, frame2, K):
    # Convert images to grayscale
    gray1 = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)

    # Detect ORB features and compute descriptors
    orb = cv2.ORB_create()
    kp1, des1 = orb.detectAndCompute(gray1, None)
    kp2, des2 = orb.detectAndCompute(gray2, None)

    # Match descriptors using BFMatcher
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches = bf.match(des1, des2)
    matches = sorted(matches, key=lambda x: x.distance)

    # Extract matched keypoints
    points1 = np.float32([kp1[m.queryIdx].pt for m in matches]).reshape(-1, 2)
    points2 = np.float32([kp2[m.trainIdx].pt for m in matches]).reshape(-1, 2)

    # Compute essential matrix
    E, mask = cv2.findEssentialMat(points2, points1, K, method=cv2.RANSAC, prob=0.999, threshold=1.0)
    _, R, t, mask = cv2.recoverPose(E, points2, points1, K)

    return R, t

# Function to process a directory of images
def process_directory(directory_path):
    # Get all image file paths from the directory
    image_paths = sorted(glob(os.path.join(directory_path, "*.png")))

    # Ensure there are images to process
    if len(image_paths) < 2:
        print("Need at least two images to compute loop closures.")
        return

    # Camera intrinsic matrix (example values, adjust to your camera)
    K = np.array([[718.856, 0, 607.1928],
                  [0, 718.856, 185.2157],
                  [0, 0, 1]])

    # Initialize ORB detector
    orb = cv2.ORB_create()

    # Initialize feature database
    features_db = []

    # Initialize poses and loop closures
    poses = []
    loop_closures = []

    # Iterate over all images
    for i in range(len(image_paths) - 1):
        frame1 = cv2.imread(image_paths[i])
        frame2 = cv2.imread(image_paths[i + 1])

        # Extract features and descriptors
        keypoints1, descriptors1 = extract_features(frame1, orb)
        keypoints2, descriptors2 = extract_features(frame2, orb)
        features_db.append(descriptors1)

        # Perform loop closure detection
        loop_idx = loop_closure_detection(features_db[:-1], descriptors2)

        # Estimate camera motion from feature extraction
        R, t = estimate_camera_motion(frame1, frame2, K)
        poses.append((R.tolist(), t.tolist()))

        # Add loop closure if detected
        if loop_idx is not None:
            loop_closures.append((i + 1, loop_idx))
            print(f"Loop closure detected between image {i + 1} and image {loop_idx}")

    # Save results to file
    results = {
        "poses": poses,
        "loop_closures": loop_closures,
        "image_paths": image_paths
    }
    with open('loop_closure_results.json', 'w') as f:
        json.dump(results, f)

    print("Loop closure detection completed and results saved to 'loop_closure_results.json'.")

# Path to the directory containing images
directory_path = '/home/thales1/VSLAM/a/pythonProject1/DATA/Data2'

# Process the directory
process_directory(directory_path)
