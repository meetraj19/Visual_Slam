"""
Complete Visual Odometry Pipeline Implementation

This module implements a visual odometry system following the visual SLAM diagram.
It handles feature extraction, matching, depth estimation, and trajectory visualization.

Key components:
1. FrameData class for organizing frame information
2. VisualOdometryPipeline class for processing image sequences
3. Utilities for visualization and data saving
"""

import cv2
import numpy as np
from pathlib import Path
from typing import List, Tuple, Dict, Optional
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from transformers import pipeline
from PIL import Image
import logging
import torch
from dataclasses import dataclass, field
import time

# Set up logging
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


@dataclass
class FrameData:
    """
    Data structure to hold frame information for better organization.
    """
    name: str
    image: np.ndarray
    keypoints: Optional[List[cv2.KeyPoint]] = None
    descriptors: Optional[np.ndarray] = None
    depth_map: Optional[np.ndarray] = None
    timestamp: float = field(default_factory=lambda: time.time())


class VisualOdometryPipeline:
    """
    Main class implementing the visual odometry pipeline.
    Handles feature extraction, matching, depth estimation, and pose calculation.
    """

    def __init__(self, camera_matrix: np.ndarray, use_gpu: bool = True,
                 min_features: int = 2000, min_matches: int = 8):
        """
        Initialize the visual odometry pipeline.

        Args:
            camera_matrix: 3x3 camera intrinsic matrix
            use_gpu: Whether to use GPU acceleration
            min_features: Minimum number of features to detect
            min_matches: Minimum number of matches required for pose estimation
        """
        self.K = camera_matrix
        self.K_inv = np.linalg.inv(camera_matrix)
        self.min_features = min_features
        self.min_matches = min_matches

        # Set up GPU if available
        self.device = 'cuda' if (use_gpu and torch.cuda.is_available()) else 'cpu'
        logger.info(f"Using device: {self.device}")

        # Initialize feature detector
        self.feature_detector = cv2.SIFT_create(
            nfeatures=min_features,
            nOctaveLayers=3,
            contrastThreshold=0.03,
            edgeThreshold=10,
            sigma=1.6
        )

        # Initialize depth estimation model
        self.depth_estimator = pipeline(
            task="depth-estimation",
            model="depth-anything/Depth-Anything-V2-Small-hf",
            device=self.device
        )
        logger.info("Depth estimation model initialized")

        # Initialize feature matcher
        FLANN_INDEX_KDTREE = 1
        index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
        search_params = dict(checks=50)
        self.matcher = cv2.FlannBasedMatcher(index_params, search_params)

        # Initialize optical flow parameters
        self.lk_params = dict(
            winSize=(21, 21),
            maxLevel=3,
            criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01),
            minEigThreshold=0.001
        )

        # Initialize trajectory storage
        self.trajectory = []
        self.current_pose = np.eye(4)

    def preprocess_image(self, image: np.ndarray) -> np.ndarray:
        """
        Preprocess image for feature detection.
        """
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()

        # Enhance contrast
        gray = cv2.equalizeHist(gray)
        return gray

    def extract_features(self, image: np.ndarray) -> Tuple[List[cv2.KeyPoint], np.ndarray]:
        """
        Extract features from an image using SIFT detector.
        """
        gray = self.preprocess_image(image)

        keypoints, descriptors = self.feature_detector.detectAndCompute(gray, None)

        if len(keypoints) < self.min_features:
            logger.warning(f"Only {len(keypoints)} features detected, attempting recovery...")
            backup_detector = cv2.SIFT_create(
                nfeatures=self.min_features * 2,
                contrastThreshold=0.02
            )
            keypoints, descriptors = backup_detector.detectAndCompute(gray, None)

        return keypoints, descriptors

    def match_features(self, kp1: List[cv2.KeyPoint], desc1: np.ndarray,
                       kp2: List[cv2.KeyPoint], desc2: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Enhanced feature matching with better point handling and validation.
        """
        if desc1 is None or desc2 is None:
            logger.warning("No descriptors available for matching")
            return np.array([]), np.array([])

        # Convert descriptors to correct format
        desc1 = np.float32(desc1)
        desc2 = np.float32(desc2)

        try:
            # Perform kNN matching
            matches = self.matcher.knnMatch(desc1, desc2, k=2)
        except Exception as e:
            logger.error(f"Feature matching failed: {str(e)}")
            return np.array([]), np.array([])

        # Apply ratio test with more careful filtering
        good_matches = []
        for match_pair in matches:
            if len(match_pair) != 2:
                continue
            m, n = match_pair
            if m.distance < 0.7 * n.distance:
                good_matches.append(m)

        if len(good_matches) < self.min_matches:
            logger.warning(f"Insufficient good matches: {len(good_matches)}")
            return np.array([]), np.array([])

        # Extract and validate matched points
        pts1 = np.float32([kp1[m.queryIdx].pt for m in good_matches])
        pts2 = np.float32([kp2[m.trainIdx].pt for m in good_matches])

        # Ensure points are in correct shape for OpenCV functions
        pts1 = pts1.reshape(-1, 2)
        pts2 = pts2.reshape(-1, 2)

        return pts1, pts2

    def reject_outliers(self, pts1: np.ndarray, pts2: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Enhanced outlier rejection with proper point validation.
        """
        if pts1.size == 0 or pts2.size == 0:
            return np.array([]), np.array([])

        if pts1.shape != pts2.shape:
            logger.error("Point arrays have different shapes")
            return np.array([]), np.array([])

        try:
            # First, use RANSAC with fundamental matrix
            F, mask = cv2.findFundamentalMat(
                pts1, pts2,
                method=cv2.RANSAC,
                ransacReprojThreshold=3,
                confidence=0.99
            )

            if mask is None:
                logger.warning("Fundamental matrix estimation failed")
                return np.array([]), np.array([])

            mask = mask.ravel().astype(bool)

            # Ensure we have enough inliers
            if np.sum(mask) < self.min_matches:
                logger.warning(f"Too few inliers after RANSAC: {np.sum(mask)}")
                return np.array([]), np.array([])

            pts1_clean = pts1[mask]
            pts2_clean = pts2[mask]

            return pts1_clean, pts2_clean

        except cv2.error as e:
            logger.error(f"RANSAC outlier rejection failed: {str(e)}")
            return np.array([]), np.array([])

    def calculate_optical_flow(self, img1: np.ndarray, img2: np.ndarray,
                               pts1: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Calculate optical flow using Lucas-Kanade method.
        """
        gray1 = self.preprocess_image(img1)
        gray2 = self.preprocess_image(img2)

        pts2, status, error = cv2.calcOpticalFlowPyrLK(
            gray1, gray2, pts1.reshape(-1, 1, 2), None, **self.lk_params
        )

        good_status = status.ravel() == 1
        return pts1[good_status], pts2[good_status], error[good_status]

    def estimate_depth(self, image: np.ndarray) -> np.ndarray:
        """
        Estimate depth map using the pre-trained model.
        """
        try:
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            image_pil = Image.fromarray(image_rgb)

            depth_output = self.depth_estimator(image_pil)
            depth_map = np.array(depth_output["depth"])

            # Normalize to metric scale
            depth_map = (depth_map - depth_map.min()) / (depth_map.max() - depth_map.min())
            depth_map = depth_map * 10.0  # Scale to approximate metric depth

            return depth_map

        except Exception as e:
            logger.error(f"Depth estimation failed: {str(e)}")
            return np.ones_like(image[:, :, 0], dtype=np.float32) * 5.0

    def estimate_relative_pose(self, pts1: np.ndarray, pts2: np.ndarray,
                               depth1: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Enhanced relative pose estimation with proper point handling and validation.
        """
        # Validate input points
        if pts1.size == 0 or pts2.size == 0:
            logger.warning("Empty point arrays in pose estimation")
            return np.eye(3), np.zeros((3, 1))

        # Ensure points are properly shaped
        pts1 = pts1.reshape(-1, 2)
        pts2 = pts2.reshape(-1, 2)

        if pts1.shape[0] < self.min_matches:
            logger.warning(f"Insufficient points for pose estimation: {pts1.shape[0]}")
            return np.eye(3), np.zeros((3, 1))

        try:
            # Essential matrix estimation with enhanced error checking
            E, mask = cv2.findEssentialMat(
                pts1, pts2, self.K,
                method=cv2.RANSAC,
                prob=0.999,
                threshold=1.0
            )

            if E is None or mask is None:
                logger.warning("Essential matrix estimation failed")
                return np.eye(3), np.zeros((3, 1))

            # Recover pose
            _, R, t, pose_mask = cv2.recoverPose(E, pts1, pts2, self.K, mask=mask)

            # Scale translation using depth information
            if depth1 is not None:
                valid_depths = []
                for pt in pts1:
                    y, x = int(pt[1]), int(pt[0])
                    if 0 <= y < depth1.shape[0] and 0 <= x < depth1.shape[1]:
                        valid_depths.append(depth1[y, x])

                if valid_depths:
                    scale = np.median(valid_depths)
                    t *= scale

            return R, t

        except cv2.error as e:
            logger.error(f"Pose estimation failed: {str(e)}")
            return np.eye(3), np.zeros((3, 1))

    def process_frame_pair(self, frame1: np.ndarray, frame2: np.ndarray) -> Dict:
        """
        Process a pair of frames through the complete pipeline.
        """
        try:
            # Feature extraction
            kp1, desc1 = self.extract_features(frame1)
            kp2, desc2 = self.extract_features(frame2)

            # Feature matching
            pts1, pts2 = self.match_features(kp1, desc1, kp2, desc2)

            # Outlier rejection
            pts1_clean, pts2_clean = self.reject_outliers(pts1, pts2)

            # Optical flow
            pts1_flow, pts2_flow, _ = self.calculate_optical_flow(frame1, frame2, pts1_clean)

            # Depth estimation
            depth1 = self.estimate_depth(frame1)

            # Pose estimation
            R, t = self.estimate_relative_pose(pts1_flow, pts2_flow, depth1)

            # Update trajectory
            T = np.eye(4)
            T[:3, :3] = R
            T[:3, 3] = t.ravel()
            self.current_pose = self.current_pose @ T
            self.trajectory.append(self.current_pose.copy())

            return {
                'success': True,
                'R': R,
                't': t,
                'matched_points': len(pts1_flow),
                'pose': self.current_pose,
                'depth_map': depth1
            }

        except Exception as e:
            logger.error(f"Frame processing failed: {str(e)}")
            return {
                'success': False,
                'error': str(e),
                'pose': self.current_pose
            }


def process_image_sequence(image_folder: str, camera_matrix: np.ndarray,
                           output_dir: Optional[str] = None,
                           use_gpu: bool = True,
                           visualize: bool = True) -> Dict:
    """
    Process a sequence of images with real-time visualization.

    Args:
        image_folder: Path to the image sequence
        camera_matrix: Camera intrinsic parameters
        output_dir: Directory to save results (optional)
        use_gpu: Whether to use GPU acceleration
        visualize: Whether to show visualizations
    """
    if output_dir:
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

    # Initialize pipeline
    vo = VisualOdometryPipeline(camera_matrix, use_gpu=use_gpu)
    logger.info("Visual odometry pipeline initialized")

    # Set up visualization
    if visualize:
        plt.ion()  # Enable interactive mode
        fig = plt.figure(figsize=(15, 5))
        ax1 = fig.add_subplot(131)  # Depth map
        ax2 = fig.add_subplot(132, projection='3d')  # Trajectory
        ax3 = fig.add_subplot(133)  # Current frame
        plt.show()

    # Load images
    image_path = Path(image_folder)
    image_files = sorted(image_path.glob('*.png'))

    if len(image_files) < 2:
        raise ValueError(f"Insufficient images found in {image_folder}")

    results = []
    processing_times = []

    for i in range(len(image_files) - 1):
        start_time = time.time()

        # Load frame pair
        frame1 = cv2.imread(str(image_files[i]))
        frame2 = cv2.imread(str(image_files[i + 1]))

        if frame1 is None or frame2 is None:
            logger.warning(f"Failed to load frames {i} or {i + 1}")
            continue

        # Process frame pair
        result = vo.process_frame_pair(frame1, frame2)
        processing_time = time.time() - start_time
        processing_times.append(processing_time)

        if result['success']:
            logger.info(f"Processed frames {i} and {i + 1} in {processing_time:.2f}s")
            results.append(result)

            # Update visualization
            if visualize:
                # Clear previous plots
                ax1.clear()
                ax2.clear()
                ax3.clear()

                # Plot depth map
                depth_map = result['depth_map']
                im1 = ax1.imshow(depth_map, cmap='plasma')
                plt.colorbar(im1, ax=ax1, label='Depth (m)')
                ax1.set_title('Depth Map')

                # Plot trajectory
                if vo.trajectory:
                    points = np.array([pose[:3, 3] for pose in vo.trajectory])
                    ax2.plot(points[:, 0], points[:, 1], points[:, 2], 'b-')
                    ax2.scatter(points[:, 0], points[:, 1], points[:, 2],
                                c='r', marker='o')
                ax2.set_xlabel('X (m)')
                ax2.set_ylabel('Y (m)')
                ax2.set_zlabel('Z (m)')
                ax2.set_title('Camera Trajectory')

                # Plot current frame
                ax3.imshow(cv2.cvtColor(frame2, cv2.COLOR_BGR2RGB))
                ax3.set_title(f'Frame {i + 1}')

                # Update display
                plt.tight_layout()
                plt.draw()
                plt.pause(0.01)

                # Save visualization if output directory is specified
                if output_dir:
                    plt.savefig(str(output_path / f"frame_{i:04d}.png"))

    # Close visualization
    if visualize:
        plt.ioff()
        plt.show()

    # Save final results
    if output_dir:
        np.save(str(output_path / "trajectory.npy"),
                [pose.tolist() for pose in vo.trajectory])

        stats = {
            'num_frames': len(image_files),
            'successful_frames': len(results),
            'average_time': np.mean(processing_times)
        }
        np.save(str(output_path / "statistics.npy"), stats)

    return {
        'trajectory': vo.trajectory,
        'statistics': {
            'num_frames': len(image_files),
            'successful_frames': len(results),
            'average_time': np.mean(processing_times)
        }
    }


if __name__ == "__main__":
    # Camera matrix
    K = np.array([
        [718.8560, 0.0, 607.1928],
        [0.0, 718.8560, 185.2157],
        [0.0, 0.0, 1.0]
    ])

    # Set up parameters
    image_folder = "/home/thales1/VSLAM/Vslam_algorithm/pythonProject1/Visual-Slam-For-Autonomous-Train-System/sample_images"  # Replace with your image folder
    output_dir = "/home/thales1/VSLAM/Vslam_algorithm/pythonProject1/Visual-Slam-For-Autonomous-Train-System/output"  # Optional: specify output directory for saving results

    try:
        # Process sequence with visualization enabled
        results = process_image_sequence(
            image_folder=image_folder,
            camera_matrix=K,
            output_dir=output_dir,
            use_gpu=True,
            visualize=True  # Enable visualization
        )

        # Print statistics
        stats = results['statistics']
        print("\nProcessing completed successfully!")
        print(f"Total frames: {stats['num_frames']}")
        print(f"Successful frames: {stats['successful_frames']}")
        print(f"Average processing time: {stats['average_time']:.2f} seconds")

    except Exception as e:
        logger.error(f"Pipeline execution failed: {str(e)}")
        import traceback

        traceback.print_exc()