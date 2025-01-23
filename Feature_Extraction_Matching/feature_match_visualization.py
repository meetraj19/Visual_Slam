import cv2
from feature_SIFT import load_images_from_folder, extract_features
from SLAM01_1.feature_match import match_features

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


def visualize_matches(images, matches, screen_size=(1366, 768)):
    """
    Visualize the matched features between consecutive image pairs.

    Args:
    - images : List of tuples (image_name, image_data).
    - matches : List of tuples (img1_name, img2_name, good_matches).
    - screen_size : Tuple containing the screen width and height (default is (1366, 768)).

    This function will display the matches between each consecutive image pair.
    """
    # Create a dictionary for quick image lookup by name
    image_dict = {name: img for name, img in images}

    for img1_name, img2_name, good_matches in matches:
        img1 = image_dict[img1_name]
        img2 = image_dict[img2_name]

        # Convert the keypoints and descriptors from the matching result
        kp1 = [kp for (name, kp, des) in features if name == img1_name][0]
        kp2 = [kp for (name, kp, des) in features if name == img2_name][0]

        # Draw the matches
        img_matches = cv2.drawMatches(img1, kp1, img2, kp2, good_matches, None,
                                      flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)

        # Resize the image to fit within the screen size
        img_matches_resized = resize_to_fit_screen(img_matches, screen_size)

        # Display the matches
        cv2.imshow(f'Matches between {img1_name} and {img2_name}', img_matches_resized)
        cv2.waitKey(0)  # Wait for a key press to show the next match

    cv2.destroyAllWindows()


# Example usage:
if __name__ == "__main__":
    # Load images
    images = load_images_from_folder(
        '/home/thales1/VSLAM/Vslam_algorithm/pythonProject1/Visual_slam_optimized/DATA/Data1')

    # Extract features from loaded images
    features = extract_features(images, visualize=False)  # No need to visualize here, we do it in matches

    # Perform feature matching on the extracted features
    matches = match_features(features)

    # Visualize the matches
    visualize_matches(images, matches)
