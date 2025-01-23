import cv2
import numpy as np
from pathlib import Path
from typing import List, Tuple, Union
import json


def load_images(source: Union[str, Path], is_video: bool = False, resize_dim: Tuple[int, int] = None) -> List[
    Tuple[str, np.ndarray]]:
    """
    Load images from a folder or video file.

    :param source: Path to the folder or video file
    :param is_video: Boolean indicating if the source is a video file
    :param resize_dim: Optional tuple to resize images (width, height)
    :return: List of tuples (image_name, image_data)
    """
    images = []
    source_path = Path(source)

    if is_video:
        cap = cv2.VideoCapture(str(source_path))
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
        for file_path in source_path.glob('*'):
            if file_path.suffix.lower() in ('.png', '.jpg', '.jpeg'):
                img = cv2.imread(str(file_path))
                if img is not None:
                    if resize_dim:
                        img = cv2.resize(img, resize_dim)
                    images.append((file_path.name, img))

    return images


def extract_orb_features(images: List[Tuple[str, np.ndarray]], num_features: int = 1000) -> List[
    Tuple[str, List[cv2.KeyPoint], np.ndarray]]:
    """
    Extract ORB features from a list of images.

    :param images: List of tuples (image_name, image_data)
    :param num_features: Number of ORB features to extract
    :return: List of tuples (image_name, keypoints, descriptors) for each image
    """
    orb = cv2.ORB_create(nfeatures=num_features)
    features = []

    for image_name, image in images:
        gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        gray_image = cv2.equalizeHist(gray_image)
        gray_image = cv2.GaussianBlur(gray_image, (5, 5), 0)

        keypoints, descriptors = orb.detectAndCompute(gray_image, None)

        if keypoints and descriptors is not None:
            features.append((image_name, keypoints, descriptors))
        else:
            print(f"No ORB keypoints detected in {image_name}")

    return features


def save_features(features: List[Tuple[str, List[cv2.KeyPoint], np.ndarray]], output_dir: Union[str, Path]):
    """
    Save extracted ORB features (keypoints and descriptors) to separate subdirectories.

    :param features: List of tuples (image_name, keypoints, descriptors)
    :param output_dir: Base directory to save the features
    """
    output_path = Path(output_dir)
    keypoints_dir = output_path / "keypoints"
    descriptors_dir = output_path / "descriptors"

    keypoints_dir.mkdir(parents=True, exist_ok=True)
    descriptors_dir.mkdir(parents=True, exist_ok=True)

    for image_name, keypoints, descriptors in features:
        base_name = Path(image_name).stem

        # Save keypoints
        keypoints_data = {
            "keypoints": [
                {
                    "x": kp.pt[0],
                    "y": kp.pt[1],
                    "size": kp.size,
                    "angle": kp.angle,
                    "response": kp.response,
                    "octave": kp.octave,
                    "class_id": kp.class_id
                } for kp in keypoints
            ]
        }
        with open(keypoints_dir / f"{base_name}_orb_keypoints.json", 'w') as f:
            json.dump(keypoints_data, f, indent=2)

        # Save descriptors
        np.save(descriptors_dir / f"{base_name}_orb_descriptors.npy", descriptors)


def main(source: Union[str, Path],
         is_video: bool = False,
         resize_dim: Tuple[int, int] = None,
         num_features: int = 1000,
         output_dir: Union[str, Path] = 'orb_features_output'):
    """
    Main function to run the ORB feature extraction pipeline.

    :param source: Path to the image folder or video file
    :param is_video: Boolean indicating if the source is a video file
    :param resize_dim: Optional tuple to resize images (width, height)
    :param num_features: Number of ORB features to extract
    :param output_dir: Base directory to save the extracted features
    """
    images = load_images(source, is_video, resize_dim)
    features = extract_orb_features(images, num_features)
    save_features(features, output_dir)
    print(f"Extracted and saved ORB features for {len(features)} images.")
    print(f"Keypoints saved in: {Path(output_dir) / 'keypoints'}")
    print(f"Descriptors saved in: {Path(output_dir) / 'descriptors'}")


if __name__ == "__main__":
    # Example usage
    main(source='/home/thales1/VSLAM/Vslam_algorithm/pythonProject1/slam01_2/data',
         is_video=False,
         resize_dim=(640, 480),
         num_features=1000,
         output_dir='orb_features_output')