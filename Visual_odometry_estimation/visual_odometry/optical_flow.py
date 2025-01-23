import cv2
import numpy as np

def calculate_optical_flow(image1, image2, keypoints1):
    """
    Calculate the optical flow using the Lucas-Kanade method.

    """
    # Convert keypoints to numpy array
    p0 = np.array([kp.pt for kp in keypoints1], dtype=np.float32).reshape(-1, 1, 2)

    # Calculate optical flow
    p1, st, err = cv2.calcOpticalFlowPyrLK(image1, image2, p0, None)

    # Select good points
    keypoints2 = p1[st == 1]
    keypoints1 = p0[st == 1]

    return keypoints1, keypoints2, st

# image1 = cv2.imread('path_to_first_image', cv2.IMREAD_GRAYSCALE)
# image2 = cv2.imread('path_to_second_image', cv2.IMREAD_GRAYSCALE)
# keypoints1 = [cv2.KeyPoint(x, y, size) for x, y, size in keypoint_data]
# keypoints1, keypoints2, status = calculate_optical_flow(image1, image2, keypoints1)
