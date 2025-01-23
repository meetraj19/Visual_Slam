import sys
from pathlib import Path

# Add the project root directory to the Python path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

import cv2
import numpy as np
from sklearn.cluster import MiniBatchKMeans
from scipy.spatial.distance import cdist
from typing import List, Tuple
import matplotlib.pyplot as plt
import logging

from Module2.visual_odometry1 import load_images, extract_features

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class LoopClosureDetector:
    def __init__(self, num_clusters=1000):
        self.num_clusters = num_clusters
        self.vocab = None
        self.kmeans = MiniBatchKMeans(n_clusters=num_clusters, random_state=0, batch_size=1000)
        self.image_histograms = []
        self.image_names = []
        self.descriptors = []

    def build_vocabulary(self, features):
        all_descriptors = np.vstack([desc for _, desc in features if desc is not None])
        self.kmeans.fit(all_descriptors)
        self.vocab = self.kmeans.cluster_centers_

    def compute_bow_histogram(self, descriptors):
        words = self.kmeans.predict(descriptors)
        hist, _ = np.histogram(words, bins=np.arange(self.num_clusters + 1), density=True)
        return hist

    def add_image(self, image_name, descriptors):
        if self.vocab is None:
            raise ValueError("Vocabulary has not been built yet")
        hist = self.compute_bow_histogram(descriptors)
        self.image_histograms.append(hist)
        self.image_names.append(image_name)
        self.descriptors.append(descriptors)
        return hist

    def detect_loop_closure(self, new_hist, new_descriptors, current_image_name, threshold=0.75):
        if len(self.image_histograms) == 0:
            return None

        dists = cdist([new_hist], self.image_histograms, 'cosine')[0]
        candidates = np.where(dists < threshold)[0]

        best_match = None
        best_inliers = 0

        for idx in candidates:
            if self.image_names[idx] == current_image_name:
                continue

            matches = self.match_descriptors(new_descriptors, self.descriptors[idx])
            if matches is not None:
                inliers = self.geometric_verification(matches, new_descriptors, self.descriptors[idx])
                if inliers > best_inliers:
                    best_inliers = inliers
                    best_match = self.image_names[idx]

        return best_match if best_inliers > 15 else None

    def match_descriptors(self, desc1, desc2):
        FLANN_INDEX_KDTREE = 1
        index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
        search_params = dict(checks=50)
        flann = cv2.FlannBasedMatcher(index_params, search_params)
        matches = flann.knnMatch(desc1, desc2, k=2)
        good_matches = []
        for m, n in matches:
            if m.distance < 0.7 * n.distance:
                good_matches.append(m)
        return good_matches if len(good_matches) > 10 else None

    def geometric_verification(self, matches, kp1, kp2):
        src_pts = np.float32([kp1[m.queryIdx] for m in matches]).reshape(-1, 1, 2)
        dst_pts = np.float32([kp2[m.trainIdx] for m in matches]).reshape(-1, 1, 2)
        _, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
        return np.sum(mask) if mask is not None else 0

def perform_loop_closure_detection(features: List[Tuple[str, List[cv2.KeyPoint], np.ndarray]],
                                   num_clusters: int = 1000,
                                   threshold: float = 0.75) -> List[Tuple[str, str]]:
    loop_closure_detector = LoopClosureDetector(num_clusters=num_clusters)
    loop_closure_detector.build_vocabulary([(name, desc) for name, _, desc in features])

    loop_closures = []

    for img_name, _, desc in features:
        if desc is not None:
            hist = loop_closure_detector.add_image(img_name, desc)
            loop_closure = loop_closure_detector.detect_loop_closure(hist, desc, img_name, threshold=threshold)
            if loop_closure:
                loop_closures.append((img_name, loop_closure))
                logger.info(f"Loop closure detected between {img_name} and {loop_closure}")
            else:
                logger.info(f"No loop closure detected for {img_name}")

    return loop_closures

def visualize_loop_closures(images: List[Tuple[str, np.ndarray]], loop_closures: List[Tuple[str, str]]):
    image_dict = {name: img for name, img in images}
    for current_image, matched_image in loop_closures:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))
        ax1.imshow(cv2.cvtColor(image_dict[current_image], cv2.COLOR_BGR2RGB))
        ax1.set_title(f"Current: {current_image}")
        ax1.axis('off')
        ax2.imshow(cv2.cvtColor(image_dict[matched_image], cv2.COLOR_BGR2RGB))
        ax2.set_title(f"Matched: {matched_image}")
        ax2.axis('off')
        plt.suptitle("Loop Closure Detection")
        plt.show()

def main(image_folder: str):
    images = load_images(Path(image_folder))
    if not images:
        logger.error(f"No images found in {image_folder}")
        return

    logger.info(f"Loaded {len(images)} images from {image_folder}")

    features = []
    for img_name, img in images:
        kp, desc = extract_features(img)
        features.append((img_name, kp, desc))

    loop_closures = perform_loop_closure_detection(features)

    logger.info(f"Detected {len(loop_closures)} loop closures:")
    for current_image, matched_image in loop_closures:
        logger.info(f"  {current_image} matches with {matched_image}")

    visualize_loop_closures(images, loop_closures)

if __name__ == "__main__":
    image_folder = r"/home/thales1/VSLAM/Vslam_algorithm/pythonProject1/The_SLAM/data"  # Update this path
    main(image_folder)