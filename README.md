VISUAL SLAM 

1. Input: RGB-D Images
The system starts with input images captured by an RGB-D camera, which provides both color (RGB) and depth (D) information for each pixel. This dual information helps in
accurately reconstructing the 3D environment.

2.  Feature Detection and Extraction
• The ﬁrst step in the pipeline is the detection of key features in the RGB images. These features serve as identiﬁable points that the system can track across multiple frames.
• Feature extraction involves identifying unique patterns or structures within the image that can be reliably matched between frames.

3.  Depth Estimation: Disparity Map
Using the depth data from the RGB-D sensor, a disparity map is generated. This map highlights the differences in position of corresponding points in the stereo images, providing
crucial depth information.
• The disparity map is used to calculate the depth of objects in the scene, which is essential for constructing a 3D map.

4. Point Cloud Generation
The features and depth information are used to generate a point cloud, which is a 3D representation of the environment. Each point in the cloud corresponds to a speciﬁc feature in
the image, with its position in 3D space determined by the depth data.

5. Camera Pose Estimation
Using the point cloud, the system estimates the pose of the camera (its position and orientation) relative to the environment. This step involves solving geometric relationships
between the points in the point cloud and their corresponding features in the image.
• Pose estimation is crucial for understanding how the camera moves through the environment and helps in localizing the camera within the map.

6. Loop Closure Detection
To correct for drift and ensure the map remains consistent over time, the system employs loop closure detection. This process identiﬁes when the camera returns to a previously
visited location, allowing the system to adjust the map and the camera's estimated trajectory to minimize errors.

7. Map Updating
As new frames are processed, the system continuously updates the map of the environment. This map is built incrementally, integrating new information from each frame to reﬁne
the 3D representation of the scene.
• The map update step ensures that the environment model remains accurate and up-to-date as the camera explores new areas or revisits known locations.

8. Output: Map and Trajectory
The ﬁnal output of the SLAM system is a detailed 3D map of the environment and the trajectory of the camera. The map provides a spatial understanding of the surroundings,
while the trajectory shows the path taken by the camera through the scene.
![Screenshot from 2024-08-12 12-48-19](https://github.com/user-attachments/assets/dc182de5-8d57-414e-8841-5e2e883ca818)
![Screenshot from 2025-01-23 15-53-09](https://github.com/user-attachments/assets/21cf0dc8-100d-4231-b71a-039c272a9852)

