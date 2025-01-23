import os
import cv2
import glob


#function to load images from a folder
def load_images_from_folder(folder):
    images = []
    for filename in glob.glob(os.path.join(folder, '*.*')):
        if filename.endswith(('.png')):
            img = cv2.imread(filename)
            if img is not None:
                images.append((filename, img))
    return images

#function to load images from a video
def load_images_from_video(video_path, resize_dim = None):
    cap = cv2.VideoCapture(video_path)
    images = []
    frame_count = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        if resize_dim:
            frame = cv2.resize(frame, resize_dim)
        images.append((f"frame_{frame_count}.png", frame))
        frame_count += 1
    cap.release()
    return images

def load_images(source , is_video = False, resize_dim = None):
    if is_video:
        return load_images_from_video(source, resize_dim)
    else:
        return load_images_from_folder(source)

def extract_features(images, num_features = 1000, resize_dim = (640, 480), visualize = False):
    """
    extract SIFT features from a list of Images

    :param images: list of tuples(image_name , image_data)
    :param num_features: number of SiFT features to extract
    :param resize_dim: tuple to resize images to a consistent dimensions(width, height)
    :param visualize: boolean to visualize the features o the image
    :return:
    - features: List of containing tuples of ( image_name , keypoints , descriptors ) for each image.
    """
    sift = cv2.SIFT_create(nfeatures = num_features)
    features = []

    for image_name , image in images:
        image_resized = cv2.resize(image, resize_dim)
        gray_image = cv2.cvtColor(image_resized, cv2.COLOR_BGR2GRAY)

        keypoints , descriptors = sift.detectAndCompute(gray_image, None)

        if keypoints is not None and descriptors is not None:
            features.append((image_name, keypoints, descriptors))

            if visualize:
                img_with_keypoints = cv2.drawKeypoints(image_resized, keypoints, None)
                cv2.imshow('Features', img_with_keypoints)
                cv2.waitKey(0)

        else:
            print(f"No keypoints detected in {image_name}")

    cv2.destroyAllWindows()
    return features

#usage

#images = load_images('/home/thales1/VSLAM/Vslam_algorithm/pythonProject1/Visual_slam_optimized/DATA/Data1')

#for video file
#images = ('/home/thales1/VSLAM/Vslam_algorithm/pythonProject1/Visual_slam_optimized/DATA/istockphoto-1273783707-640_adpp_is.mp4', is_video = True, resize_dim = (640, 80))

#features= extract_features(images, visualize=True)









