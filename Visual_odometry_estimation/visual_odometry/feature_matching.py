import cv2

def match_features(directory, features):
    """
    Match features between consecutive images in a directory.

    """
    # Initialize the FLANN-based Matcher
    index_params = dict(algorithm=1, trees = 5)
    search_params = dict(checks = 50)
    flann = cv2.FlannBasedMatcher(index_params, search_params)


    all_matches = []
    for i in range(len(features) - 1):
        img1_name, kp1, des1 = features[i]
        img2_name, kp2, des2 = features[i + 1]

        # Perform KNN matching
        knn_matches = flann.knnMatch(des1, des2, k=2)

        # Apply the ratio test
        good_matches = []
        for m, n in knn_matches:
            if m.distance < 0.75 * n.distance:
                good_matches.append(m)

        all_matches.append((img1_name, img2_name, good_matches))

    return all_matches


#directory = '/home/thales1/VSLAM/Vslam_algorithm/pythonProject1/Visual_slam_optimized/DATA/Data1'
#features = extract_features(directory)
#matches = match_features(directory, features)
