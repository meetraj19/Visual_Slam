import numpy as np
from sklearn.cluster import MiniBatchKMeans
from scipy.spatial.distance import cdist
from feature_SIFT import load_images_from_folder, extract_features

class LoopClosureDetector:
    def __init__(self, num_clusters = 1000):
        self.num_clusters = num_clusters
        self.vocab = None
        self.kmeans = MiniBatchKMeans(n_clusters = num_clusters, random_state=0)
        self.descriptor_list = []
        self.image_histograms = []

    def build_vocabulary(self, features):
        """
        Build a visual vocabulary using KMeans clustering
        :param features:
        :return:
        """

        all_descriptors = np.vstack([desc for _, _, desc in features if desc is not None])

        # Fit the Kmeans model

        self.kmeans.fit(all_descriptors)
        self.vocab = self.kmeans.cluster_centers_

    def compute_bow_histogram(self, descriptors):

        """
        compute the Bag of Words histogram for a set of descriptors
        :return:
        """

        words = self.kmeans.predict(descriptors)
        hist, _ = np.histogram(words, bins = np.arange(self.num_clusters +1), density = True)
        return hist
    def add_image(self, descriptors):

        """
        Add an image descriptors to the list and compute its BoW histogram
        :param descriptors:
        :return:
        """
        if self.vocab is None:
            raise ValueError("Vocabulary has not been built yet")

        hist = self.compute_bow_histogram(descriptors)
        self.image_histograms.append(hist)
        return hist


    def detect_loop_closure(self, new_hist , threshold = 0.75):

        if len(self.image_histograms) == 0:
            return -1

        dists = cdist([new_hist], self.image_histograms, 'cosine')[0]
        idx = np.argmin(dists)
        if dists[idx] < threshold:
            return idx
        else:
            return -1

#usage

if __name__ == "__main__":
    # Load images
    image_folder = '/home/thales1/VSLAM/Vslam_algorithm/pythonProject1/Visual_slam_optimized/DATA/Data2'
    images = load_images_from_folder(image_folder)

    # Extract features from loaded images
    features = extract_features(images, visualize=False)

    # Initialize the Loop Closure Detector
    loop_closure_detector = LoopClosureDetector(num_clusters=1000)

    # Build vocabulary with initial set of features
    loop_closure_detector.build_vocabulary(features)

    # Simulate loop closure detection
    for img_name, kp, desc in features:
        if desc is not None:
            # Compute BoW histogram for the image
            hist = loop_closure_detector.add_image(desc)

            # Check for loop closure
            idx = loop_closure_detector.detect_loop_closure(hist, threshold=0.75)
            if idx != -1:
                print(f"Loop closure detected with image index {idx} for current image {img_name}")
            else:
                print(f"No loop closure detected for current image {img_name}")

