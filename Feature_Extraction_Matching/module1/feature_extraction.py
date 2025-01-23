import cv2
import numpy as np
from pathlib import Path
from typing import List, Tuple, Union, Optional
import json
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def load_images(source: Union[str, Path], is_video: bool = False, resize_dim: Optional[Tuple[int, int]] = None) -> List[
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

    if not source_path.exists():
        raise FileNotFoundError(f"Source path does not exist: {source_path}")

    if is_video:
        if not source_path.is_file():
            raise ValueError(f"Video source must be a file: {source_path}")
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
        if not source_path.is_dir():
            raise ValueError(f"Image source must be a directory: {source_path}")
        for file_path in source_path.glob('*'):
            if file_path.suffix.lower() in ('.png', '.jpg', '.jpeg'):
                img = cv2.imread(str(file_path))
                if img is not None:
                    if resize_dim:
                        img = cv2.resize(img, resize_dim)
                    images.append((file_path.name, img))
                else:
                    logging.warning(f"Failed to load image: {file_path}")

    if not images:
        raise ValueError(f"No images found in the source: {source_path}")

    return images

def extract_features(image: np.ndarray, method: str = 'SIFT', num_features: int = 1000) -> Tuple[
    List[cv2.KeyPoint], np.ndarray]:
    """
    Extract features from an image using the specified method.
    """
    if method.upper() == 'SIFT':
        feature_extractor = cv2.SIFT_create(nfeatures=num_features)
    elif method.upper() == 'ORB':
        feature_extractor = cv2.ORB_create(nfeatures=num_features)
    else:
        raise ValueError(f"Unsupported feature extraction method: {method}")

    keypoints, descriptors = feature_extractor.detectAndCompute(image, None)
    return keypoints, descriptors

def save_features(features: List[Tuple[str, List[cv2.KeyPoint], np.ndarray]], output_dir: Union[str, Path], method: str):
    """
    Save extracted features (keypoints and descriptors) to separate subdirectories.

    :param features: List of tuples (image_name, keypoints, descriptors)
    :param output_dir: Base directory to save the features
    :param method: Feature extraction method used ('ORB' or 'SIFT')
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
        with open(keypoints_dir / f"{base_name}_{method.lower()}_keypoints.json", 'w') as f:
            json.dump(keypoints_data, f, indent=2)

        # Save descriptors
        np.save(descriptors_dir / f"{base_name}_{method.lower()}_descriptors.npy", descriptors)

def main(source: Union[str, Path],
         is_video: bool = False,
         resize_dim: Optional[Tuple[int, int]] = None,
         num_features: int = 1000,
         method: str = 'ORB',
         output_dir: Union[str, Path] = 'features_output'):
    """
    Main function to run the feature extraction pipeline.

    :param source: Path to the image folder or video file
    :param is_video: Boolean indicating if the source is a video file
    :param resize_dim: Optional tuple to resize images (width, height)
    :param num_features: Number of features to extract
    :param method: Feature extraction method ('ORB' or 'SIFT')
    :param output_dir: Base directory to save the extracted features
    """
    try:
        images = load_images(source, is_video, resize_dim)
        features = extract_features(images, method, num_features)
        save_features(features, output_dir, method)
        logging.info(f"Extracted and saved {method} features for {len(features)} images.")
        logging.info(f"Keypoints saved in: {Path(output_dir) / 'keypoints'}")
        logging.info(f"Descriptors saved in: {Path(output_dir) / 'descriptors'}")
    except Exception as e:
        logging.error(f"An error occurred: {str(e)}")

if __name__ == "__main__":
    # Example usage
    main(source='/home/thales1/VSLAM/Vslam_algorithm/pythonProject1/The_SLAM/image_1',
         is_video=False,
         resize_dim=(640, 480),
         num_features=1000,
         method='SIFT',
         output_dir='features_output')