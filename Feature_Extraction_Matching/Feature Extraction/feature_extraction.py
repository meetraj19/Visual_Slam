import os
import cv2
import numpy as np
from pathlib import Path
from typing import List, Tuple, Optional


def load_images(source: str, is_video: bool = False, resize_dim: Optional[Tuple[int, int]] = None) -> List[
    Tuple[str, np.ndarray]]:
    """
    Load images from a folder or video file.

    :param source: Path to the folder or video file
    :param is_video: Boolean indicating if the source is a video file
    :param resize_dim: Optional tuple to resize images (width, height)
    :return: List of tuples (image_name, image_data)
    """
    images = []

    if is_video:
        cap = cv2.VideoCapture(source)
        frame_count = 0
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            if resize_dim:
                frame = cv2.resize(frame, resize_dim)
            images.append((f"frame_{frame_count:04d}.png", frame))
            frame_count += 1
        cap.release()
    else:
        for file_path in Path(source).glob('*'):
            if file_path.suffix.lower() in ('.png', '.jpg', '.jpeg'):
                img = cv2.imread(str(file_path))
                if img is not None:
                    if resize_dim:
                        img = cv2.resize(img, resize_dim)
                    images.append((file_path.name, img))

    return images


def extract_features(images: List[Tuple[str, np.ndarray]],
                     num_features: int = 1000,
                     resize_dim: Optional[Tuple[int, int]] = None,
                     visualize: bool = False) -> List[Tuple[str, List[cv2.KeyPoint], np.ndarray]]:
    """
    Extract SIFT features from a list of images.

    :param images: List of tuples (image_name, image_data)
    :param num_features: Number of SIFT features to extract
    :param resize_dim: Optional tuple to resize images (width, height)
    :param visualize: Boolean to visualize the features on the image
    :return: List of tuples (image_name, keypoints, descriptors) for each image
    """
    sift = cv2.SIFT_create(nfeatures=num_features)
    features = []

    for image_name, image in images:
        if resize_dim:
            image = cv2.resize(image, resize_dim)

        gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # Apply histogram equalization to improve contrast
        gray_image = cv2.equalizeHist(gray_image)

        # Apply Gaussian blur to reduce noise
        gray_image = cv2.GaussianBlur(gray_image, (5, 5), 0)

        keypoints, descriptors = sift.detectAndCompute(gray_image, None)

        if keypoints and descriptors is not None:
            features.append((image_name, keypoints, descriptors))

            if visualize:
                img_with_keypoints = cv2.drawKeypoints(image, keypoints, None,
                                                       flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
                cv2.imshow('Features', img_with_keypoints)
                cv2.waitKey(0)
        else:
            print(f"No keypoints detected in {image_name}")

    if visualize:
        cv2.destroyAllWindows()

    return features


def main():
    # Example usage
    source = '/home/thales1/VSLAM/Vslam_algorithm/pythonProject1/Visual_slam_optimized/DATA/Data2'
    is_video = False  # Set to True if using a video file
    resize_dim = (640, 480)

    images = load_images(source, is_video, resize_dim)
    features = extract_features(images, num_features=1000, visualize=True)

    print(f"Extracted features from {len(features)} images")


if __name__ == "__main__":
    main()