# SLAM Feature Detection

This repository contains Python scripts for SIFT and ORB feature detection, essential for Visual SLAM pipelines.

## Requirements

- Python 3.6+
- OpenCV (cv2)
- NumPy

Install requirements:
```
pip install opencv-python numpy
```

## Scripts

1. `sift_feature_extraction.py`: SIFT feature detection and extraction
2. `orb_feature_extraction.py`: ORB feature detection and extraction

## Usage

Command line:
```
python sift_feature_extraction.py
```
or
```
python orb_feature_extraction.py
```

In Python script:
```python
from sift_feature_extraction import main as sift_main
# or
from orb_feature_extraction import main as orb_main

sift_main(source='/path/to/images_or_video', is_video=False, resize_dim=(640, 480), num_features=1000, output_dir='sift_output')
# or
orb_main(source='/path/to/images_or_video', is_video=False, resize_dim=(640, 480), num_features=1000, output_dir='orb_output')
```

## Output

Scripts create two subdirectories in the specified output directory:
- `keypoints/`: JSON files with keypoint information
- `descriptors/`: NumPy (.npy) files with descriptor data

## SIFT vs ORB

- SIFT: More robust, higher quality, computationally expensive
- ORB: Faster, free to use, less robust to scale changes

Choose based on your specific requirements for speed, feature quality, and licensing.

## Integration

Load features in your SLAM pipeline:

```python
import json
import numpy as np

def load_features(image_name, algorithm, base_dir):
    with open(f"{base_dir}/keypoints/{image_name}_{algorithm}_keypoints.json", 'r') as f:
        keypoints_data = json.load(f)
    descriptors = np.load(f"{base_dir}/descriptors/{image_name}_{algorithm}_descriptors.npy")
    return keypoints_data, descriptors

# Usage
keypoints, descriptors = load_features("frame_0000.png", "sift", "sift_output")
```

Adjust subsequent SLAM steps based on the chosen feature detection method.