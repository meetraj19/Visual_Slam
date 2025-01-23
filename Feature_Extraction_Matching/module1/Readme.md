# Feature Extraction Module

## Overview
This module extracts ORB or SIFT features from images or video frames for use in Visual SLAM systems.

## Requirements
- Python 3.6+
- OpenCV
- NumPy

Install dependencies:
```
pip install opencv-python numpy
```

## Usage
Run the script:
```
python feature_extraction.py
```

Or import in your project:

```python
from feature_extraction import main as extract_features

extract_features(source='/path/to/data',
                 is_video=False,
                 resize_dim=(640, 480),
                 num_features=1000,
                 method='ORB',
                 output_dir='features_output')
```

## Parameters
- `source`: Path to image directory or video file
- `is_video`: True if source is a video, False for image directory
- `resize_dim`: (width, height) to resize images (optional)
- `num_features`: Number of features to extract per image
- `method`: 'ORB' or 'SIFT'
- `output_dir`: Directory to save extracted features

## Output
Saves keypoints (JSON) and descriptors (NumPy) in subdirectories of `output_dir`.

## Customization
Modify `extract_features()` function to add new extraction methods or adjust preprocessing steps.