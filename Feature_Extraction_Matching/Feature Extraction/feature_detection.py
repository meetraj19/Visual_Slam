import cv2
import numpy as np
import glob
import os
import json
import matplotlib.pyplot as plt


def detect_and_save_features(image_dir, output_keypoints_json, output_matches_json):
    # Initialize ORB detector
    orb = cv2.ORB_create()

    # Initialize Matcher
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

    # Placeholder for previous descriptors and keypoints
    prev_descriptors = None
    prev_keypoints = None
    prev_image = None
    prev_image_index = None

    # List of images
    image_paths = sorted(glob.glob(os.path.join(image_dir, '*.png')))

    # Check if images are found
    if not image_paths:
        raise ValueError("No .png images found in the directory.")

    # Dictionaries to store keypoints and matches
    keypoints_data = {"images": []}
    matches_data = []

    # Loop through all image paths
    for idx, image_path in enumerate(image_paths):
        try:
            # Load image
            current_image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
            if current_image is None:
                raise ValueError(f"Image {image_path} could not be loaded.")

            # Detect ORB features and compute descriptors
            current_keypoints, current_descriptors = orb.detectAndCompute(current_image, None)

            if current_keypoints is None or current_descriptors is None:
                print(f"No keypoints or descriptors found in image {idx}.")
                continue

            keypoints_image_data = {"image_index": idx, "keypoints": []}

            # Save keypoints to the dictionary
            for keypoint in current_keypoints:
                keypoint_data = {"x": keypoint.pt[0], "y": keypoint.pt[1]}
                keypoints_image_data["keypoints"].append(keypoint_data)

            keypoints_data["images"].append(keypoints_image_data)

            # If not the first image, match descriptors
            if prev_descriptors is not None:
                # Match descriptors with the previous image
                matches = bf.match(prev_descriptors, current_descriptors)
                matches = sorted(matches, key=lambda x: x.distance)

                match_data = {
                    "image_index_1": prev_image_index,
                    "image_index_2": idx,
                    "matches": []
                }

                # Save matches to the dictionary
                for match in matches:
                    match_data["matches"].append({
                        "queryIdx": match.queryIdx,
                        "trainIdx": match.trainIdx,
                        "distance": match.distance
                    })

                matches_data.append(match_data)

                # Draw lines between matching features
                match_img = cv2.drawMatches(prev_image, prev_keypoints, current_image, current_keypoints, matches[:50],
                                            None, flags=2, matchColor=(0, 255, 255), matchesThickness=3)

                # Convert BGR to RGB for matplotlib
                match_img = cv2.cvtColor(match_img, cv2.COLOR_BGR2RGB)

                # Show the matches
                plt.figure(figsize=(12, 6))
                plt.imshow(match_img)
                plt.title(f'Matches between image {prev_image_index} and image {idx}')
                plt.axis('off')
                plt.show()

            # Update previous image keypoints and descriptors
            prev_keypoints = current_keypoints
            prev_descriptors = current_descriptors
            prev_image = current_image
            prev_image_index = idx

        except Exception as e:
            print(f"An error occurred while processing image {idx}: {e}")

    # Save keypoints data to JSON file
    with open(output_keypoints_json, 'w') as keypoints_json_file:
        json.dump(keypoints_data, keypoints_json_file, indent=4)

    # Save matches data to JSON file
    with open(output_matches_json, 'w') as matches_json_file:
        json.dump(matches_data, matches_json_file, indent=4)


# Directory where images are stored
image_dir = '/home/thales1/VSLAM/a/pythonProject1/DATA/Data2'
output_keypoints_json = 'keypoints.json'
output_matches_json = 'matches.json'

# Run the feature detection and save keypoints and matches
detect_and_save_features(image_dir, output_keypoints_json, output_matches_json)
