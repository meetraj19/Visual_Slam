import cv2
from feature_SIFT import extract_features, load_images_from_folder


def match_features(features , ratio_thresh = 0.75):
    """
    Match features using FLANN based Matcher
    :param features: tuple
    :param threshold: ratio threshold for filtering matches
    :return: list of tuples (img_1 ,img_2 , good_matches) for each image pair

    """
    #initialize Flann matcher
    index_params = dict(algorithm = 1, trees = 5)
    search_params = dict(checks = 50)
    flann = cv2.FlannBasedMatcher(index_params, search_params)

    all_matches = []

    for i in range(len(features)-1):
        img1_name , kp1, des1 = features[i]
        img2_name, kp2, des2 = features[i+1]

        if des1 is None or des2 is None:
            print(f"Skipping matching between {img1_name} and {img2_name} due to missing descriptors.")
            continue
        #perform knn matching
        knn_matches = flann.knnMatch(des1, des2, k=2)

        #apply ratio test
        good_matches = [m for m , n in knn_matches if m.distance < ratio_thresh * n.distance]

        print(f"Matched {len(good_matches)} features between {img1_name} and {img2_name}")

        all_matches.append((img1_name, img2_name, good_matches))
    return all_matches

if __name__ == "__main__":

    images = load_images_from_folder("/home/thales1/VSLAM/Vslam_algorithm/pythonProject1/Visual_slam_optimized/DATA/Data1")

    features = extract_features(images, visualize= True)

    matches = match_features(features)