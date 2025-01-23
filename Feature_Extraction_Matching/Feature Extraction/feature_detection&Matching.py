import cv2
import numpy as np
import os
from glob import glob
import json


def detect_and_match_features(image1, image2):
    sift = cv2.SIFT_create()
    keypoints1, descriptors1 = sift.detectAndCompute(image1, None)
    keypoints2, descriptors2 = sift.detectAndCompute(image2, None)
    bf = cv2.BFMatcher(cv2.NORM_L2)
    raw_matches = bf.knnMatch(descriptors1, descriptors2, k=2)

    # Apply Lowe's ratio test
    matches = []
    for m, n in raw_matches:
        if m.distance < 0.75 * n.distance:
            matches.append(m)

    matched_image = cv2.drawMatches(image1, keypoints1, image2, keypoints2, matches[:100], None,
                                    flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
    return matched_image, keypoints1, keypoints2, matches, descriptors1, descriptors2


def save_keypoints(keypoints, frame_index):
    keypoints_filename = f"keypoints_frame_{frame_index}.json"
    data = {
        "keypoints": [{
            "Keypoint_ID": idx,
            "Image_ID": frame_index,
            "X": kp.pt[0],
            "Y": kp.pt[1],
            "Size": kp.size,
            "Angle": kp.angle
        } for idx, kp in enumerate(keypoints)]
    }
    with open(keypoints_filename, 'w') as file:
        json.dump(data, file, indent=4)


def save_descriptors(descriptors, frame_index):
    descriptors_filename = f"descriptors_frame_{frame_index}.json"
    data = {
        "descriptors": descriptors.tolist()
    }
    with open(descriptors_filename, 'w') as file:
        json.dump(data, file, indent=4)


def save_matches(matches, frame_index):
    matches_filename = f"matches_frame_{frame_index}.json"
    matches_data = {
        "matches": [{
            "Match_ID": idx,
            "Keypoint_ID_Img1": match.queryIdx,
            "Keypoint_ID_Img2": match.trainIdx
        } for idx, match in enumerate(matches)]
    }
    with open(matches_filename, 'w') as file:
        json.dump(matches_data, file, indent=4)


def load_images(folder):
    images = []
    for filename in sorted(glob(os.path.join(folder, '*.png'))):
        img = cv2.imread(filename, cv2.IMREAD_GRAYSCALE)
        if img is not None:
            images.append(img)
    return images


def load_video(video_path):
    images = []
    cap = cv2.VideoCapture(video_path)
    while cap.isOpened():
        ret, frame = cap.read()
        if ret:
            gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            images.append(gray_frame)
        else:
            break
    cap.release()
    return images


path = '/home/thales1/VSLAM01/b/pythonProject1/Data/istockphoto-1427900760-640_adpp_is.mp4'  # Use relative path

if os.path.isdir(path):
    images = load_images(path)
elif os.path.isfile(path) and path.endswith(('.mp4', '.avi')):
    images = load_video(path)
else:
    raise ValueError('Provided path is neither images nor video')

for i in range(len(images) - 1):
    matched_image, keypoints1, keypoints2, matches, descriptors1, descriptors2 = detect_and_match_features(images[i], images[i + 1])
    cv2.imshow(f'Matched Features between frame {i} and {i + 1}', matched_image)
    cv2.waitKey(0)
    save_keypoints(keypoints1, i)
    save_keypoints(keypoints2, i + 1)
    save_descriptors(descriptors1, i)
    save_descriptors(descriptors2, i + 1)
    save_matches(matches, i)

cv2.destroyAllWindows()
