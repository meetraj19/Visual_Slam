import torch
import cv2
import numpy as np
from torchvision.transforms import Compose, Normalize, Resize, ToTensor
from PIL import Image
from transformers import pipeline


def load_depth_anything_model():
    """
    Load the Depth Anything model using Hugging Face pipeline.

    Returns:
    - depth_pipe: The Hugging Face depth estimation pipeline.
    """
    depth_pipe = pipeline(task="depth-estimation", model="depth-anything/Depth-Anything-V2-Small-hf")
    return depth_pipe


def estimate_depth_depth_anything(image, depth_pipe):
    """
    Estimate depth using Depth Anything model.

    Args:
    - image (numpy.ndarray): Input RGB image.
    - depth_pipe: Loaded depth estimation pipeline from Hugging Face.

    Returns:
    - depth (numpy.ndarray): Estimated depth map.
    """
    img_pil = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

    # Perform depth estimation using the pipeline
    depth = depth_pipe(img_pil)["depth"]

    # Convert the depth map to a NumPy array
    depth_map = np.array(depth)

    # Normalize the depth map to the range [0, 1] for visualization or further processing
    depth_normalized = (depth_map - depth_map.min()) / (depth_map.max() - depth_map.min())

    return depth_normalized


# Example usage:
if __name__ == "__main__":
    # Load the Depth Anything model
    depth_pipe = load_depth_anything_model()

    # Load an image using OpenCV
    image = cv2.imread('/home/thales1/VSLAM/Vslam_algorithm/pythonProject1/Visual_slam_optimized/DATA/Screenshot from 2024-08-12 16-52-50.png')

    # Estimate depth
    depth = estimate_depth_depth_anything(image, depth_pipe)

    # Display the depth map using OpenCV or Matplotlib
    cv2.imshow('Depth Map', depth)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
