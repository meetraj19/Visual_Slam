import cv2
import numpy as np
from feature_SIFT import extract_features, load_images_from_folder
from SLAM01_1.feature_match import match_features
from Reject_outlier import reject_outliers_ransac


def optical_flow(img1, img2, kp1, kp2, good_matches):

    """
    Calculate optical flow using the Lucas-Kanade method for matched keypoints.

    Args:
    - img1, img2 : The two images between which to calculate the flow.
    - kp1, kp2 : Keypoints from the two images.
    - good_matches : List of good matches after outlier rejection.

    Returns:
    - pts1 : Points in the first image.
    - pts2 : Corresponding points in the second image with calculated optical flow.
    - status : List indicating whether flow was found for each point.
    """
    try:
        # Extract the matching keypoints' coordinates
        pts1 = np.float32([kp1[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
        pts2 = np.float32([kp2[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)
    except IndexError as e:
        print(f"IndexError: {e}")
        print(f"len(kp1) = {len(kp1)}, len(kp2) = {len(kp2)}, len(good_matches) = {len(good_matches)}")
        return None, None, None

    # Calculate optical flow using the Lucas-Kanade method
    pts2_flow, status, _ = cv2.calcOpticalFlowPyrLK(img1, img2, pts1, None, winSize=(21, 21), maxLevel=3,
                                                    criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01))

    # Filter out points where flow was not found
    pts1_flow = pts1[status == 1]
    pts2_flow = pts2_flow[status == 1]

    return pts1_flow, pts2_flow, status

def visualize_optical_flow(img1, pts1, pts2, status, screen_size=(1366, 768)):
    """
    Visualize the optical flow vectors between two images.

    Args:
    - img1 : The first image.
    - pts1 : Points in the first image.
    - pts2 : Corresponding points in the second image after optical flow.
    - status : List indicating whether flow was found for each point.
    - screen_size : Tuple containing the screen width and height (default is (1366, 768)).
    """
    # Check if the image is grayscale (single channel) or color (three channels)
    if len(img1.shape) == 2 or img1.shape[2] == 1:
        # If grayscale, convert to BGR for visualization
        img1_color = cv2.cvtColor(img1, cv2.COLOR_GRAY2BGR)
    else:
        # If already in color, use it directly
        img1_color = img1

    # Draw the flow vectors
    for (p1, p2) in zip(pts1, pts2):
        x1, y1 = p1.ravel()
        x2, y2 = p2.ravel()
        cv2.arrowedLine(img1_color, (int(x1), int(y1)), (int(x2), int(y2)), color=(0, 255, 0), thickness=2, tipLength=0.5)

    # Resize the image to fit within the screen size
    img_flow_resized = resize_to_fit_screen(img1_color, screen_size)

    # Display the optical flow
    cv2.imshow('Optical Flow', img_flow_resized)
    cv2.waitKey(0)  # Wait for a key press before closing

    cv2.destroyAllWindows()


def resize_to_fit_screen(image, screen_size = (1366, 768)):

    screen_width, screen_height = screen_size
    img_height, img_width = image.shape[:2]

    scaling_factor = min(screen_width/img_width, screen_height/img_height)

    new_size = (int(img_width*scaling_factor), int(img_height*scaling_factor))
    resized_image = cv2.resize(image, new_size, interpolation=cv2.INTER_AREA)

    return resized_image

if __name__ == "__main__":
    # Load images
    images = load_images_from_folder('/home/thales1/VSLAM/Vslam_algorithm/pythonProject1/Visual_slam_optimized/DATA/Data1')

    # Extract features from loaded images
    features = extract_features(images, visualize=False)

    # Perform feature matching on the extracted features
    matches = match_features(features)

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

        # Calculate optical flow using the good inlier matches
        pts1_flow, pts2_flow, status = optical_flow(img1, img2, kp1, kp2, good_matches_inliers)

        if pts1_flow is None or pts2_flow is None:
            print(f"Optical flow calculation failed for {img1_name} and {img2_name}.")
            continue

        # Visualize the optical flow
        visualize_optical_flow(img1, pts1_flow, pts2_flow, status)
