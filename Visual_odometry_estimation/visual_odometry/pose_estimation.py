import cv2
import numpy as np

def estimate_pose(keypoints1, keypoints2, matches, depth1, camera_matrix):
    """
    Estimate the pose between two sets of matched keypoints using the PnP algorithm.
    """
    # Extract matched keypoints
    matched_points1 = np.float32([keypoints1[m.queryIdx].pt for m in matches])
    matched_points2 = np.float32([keypoints2[m.trainIdx].pt for m in matches])

    # Get corresponding 3D points
    points_3d = []
    points_2d = []
    for pt1, pt2 in zip(matched_points1, matched_points2):
        z = depth1[int(pt1[1]), int(pt1[0])]
        if z > 0:
            x = (pt1[0] - camera_matrix[0, 2]) * z / camera_matrix[0, 0]
            y = (pt1[1] - camera_matrix[1, 2]) * z / camera_matrix[1, 1]
            points_3d.append([x, y, z])
            points_2d.append(pt2)

    points_3d = np.array(points_3d)
    points_2d = np.array(points_2d)

    # Estimate pose using PnP
    _, rvec, tvec, _ = cv2.solvePnPRansac(points_3d, points_2d, camera_matrix, None)

    return rvec, tvec
