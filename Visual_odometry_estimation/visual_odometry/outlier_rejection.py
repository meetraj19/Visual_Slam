import cv2
import numpy as np

def reject_outliers(matches, keypoints1, keypoints2):
    """
    Reject outlier matches using RANSAC.

    """
    if len(matches) < 4:
        return [], None

    # Extract location of good matches
    src_pts = np.float32([keypoints1[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
    dst_pts = np.float32([keypoints2[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)

    # Find homography matrix with RANSAC
    homography, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
    matches_mask = mask.ravel().tolist()

    good_matches = [m for m, mask in zip(matches, matches_mask) if mask]

    return good_matches, homography

