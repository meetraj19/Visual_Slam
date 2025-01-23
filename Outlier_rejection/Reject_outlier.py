import cv2
import numpy as np
from feature_SIFT import extract_features , load_images_from_folder
from SLAM01_1.feature_match import match_features


def reject_outliers_ransac(kp1, kp2, matches, reproj_thresh=3.0):
    """
    Apply RANSAC to reject outliers from the matches.

    Args:
    - kp1 : List of keypoints from the first image.
    - kp2 : List of keypoints from the second image.
    - matches : List of DMatch objects containing good matches between kp1 and kp2.
    - reproj_thresh : Reprojection threshold for RANSAC.

    Returns:
    - good_matches : List of inlier matches after outlier rejection.
    - mask : The mask of inliers as output by RANSAC.
    - homography : The homography matrix computed by RANSAC (or None if not found).
    """
    if len(matches) < 4:
        # Not enough matches to compute a reliable homography
        return [], None, None

    # Extract the matching keypoints' coordinates
    pts1 = np.float32([kp1[m.queryIdx].pt for m in matches])
    pts2 = np.float32([kp2[m.trainIdx].pt for m in matches])

    # Compute the homography using RANSAC
    homography, mask = cv2.findHomography(pts1, pts2, cv2.RANSAC, reproj_thresh)

    if homography is None:
        return [], None, None

    # Filter out outliers using the mask
    good_matches = [matches[i] for i in range(len(matches)) if mask[i]]

    return good_matches, mask, homography


def visualize_matches_with_outliers(images, matches, inliers, screen_size=(1366, 768)):
    """
    Visualize the matched features between consecutive image pairs with inliers highlighted.

    Args:
    - images : List of tuples (image_name, image_data).
    - matches : List of tuples (img1_name, img2_name, good_matches).
    - inliers : List of tuples containing the inlier matches.
    - screen_size : Tuple containing the screen width and height (default is (1366, 768)).

    This function will display the matches between each consecutive image pair with inliers highlighted.
    """
    # Create a dictionary for quick image lookup by name
    image_dict = {name: img for name, img in images}

    for (img1_name, img2_name, good_matches), inlier_mask in zip(matches, inliers):
        img1 = image_dict[img1_name]
        img2 = image_dict[img2_name]

        # Convert the keypoints and descriptors from the matching result
        kp1 = [kp for (name, kp, des) in features if name == img1_name][0]
        kp2 = [kp for (name, kp, des) in features if name == img2_name][0]

        # Ensure inlier_mask is a list of integers
        inlier_mask = inlier_mask.ravel().tolist()

        # Draw the matches
        img_matches = cv2.drawMatches(img1, kp1, img2, kp2, good_matches, None, matchesMask=inlier_mask, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)

        # Resize the image to fit within the screen size
        img_matches_resized = resize_to_fit_screen(img_matches, screen_size)

        # Display the matches
        cv2.imshow(f'Matches between {img1_name} and {img2_name} with RANSAC Inliers', img_matches_resized)
        cv2.waitKey(0)  # Wait for a key press to show next match

    cv2.destroyAllWindows()

def resize_to_fit_screen(image, screen_size=(1366, 768)):
    """
    Resize the image to fit within the specified screen size while maintaining aspect ratio.

    Args:
    - image : The image to resize.
    - screen_size : Tuple containing the screen width and height (default is (1366, 768)).

    Returns:
    - Resized image.
    """
    screen_width, screen_height = screen_size
    img_height, img_width = image.shape[:2]

    # Calculate the scaling factor to fit the image within the screen size
    scaling_factor = min(screen_width / img_width, screen_height / img_height)

    # Resize the image using the scaling factor
    new_size = (int(img_width * scaling_factor), int(img_height * scaling_factor))
    resized_image = cv2.resize(image, new_size, interpolation=cv2.INTER_AREA)

    return resized_image


# Example usage:
if __name__ == "__main__":
    # Load images
    images = load_images_from_folder(
        '/home/thales1/VSLAM/Vslam_algorithm/pythonProject1/Visual_slam_optimized/DATA/Data1')

    # Extract features from loaded images
    features = extract_features(images, visualize=False)  # No need to visualize here, we do it in matches

    # Perform feature matching on the extracted features
    matches = match_features(features)

    inliers = []

    # Outlier rejection with RANSAC for each pair of matches
    for i, (img1_name, img2_name, good_matches) in enumerate(matches):
        kp1 = [kp for (name, kp, des) in features if name == img1_name][0]
        kp2 = [kp for (name, kp, des) in features if name == img2_name][0]

        # Apply RANSAC to reject outliers
        good_matches_inliers, inlier_mask, _ = reject_outliers_ransac(kp1, kp2, good_matches)
        inliers.append(inlier_mask)

        print(
            f"Retained {len(good_matches_inliers)} inliers after RANSAC out of {len(good_matches)} matches between {img1_name} and {img2_name}")

    # Visualize the matches with inliers highlighted
    visualize_matches_with_outliers(images, matches, inliers)
