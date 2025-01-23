import numpy as np
import open3d as o3d

def create_point_cloud(image, depth, fx, fy, cx, cy):
    """
    Create a point cloud from a single image and depth map.

    """
    height, width = depth.shape
    mask = depth > 0

    u, v = np.meshgrid(np.arange(width), np.arange(height))
    u = u[mask]
    v = v[mask]
    z = depth[mask]

    x = (u - cx) * z / fx
    y = (v - cy) * z / fy

    points = np.vstack((x, y, z)).T

    colors = image[mask]
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    pcd.colors = o3d.utility.Vector3dVector(colors.astype(np.float64) / 255.0)

    return pcd

def integrate_local_map(images, depth_maps, fx, fy, cx, cy):
    """
    Integrate local maps into a global point cloud map.

    """
    global_map = o3d.geometry.PointCloud()

    for image, depth in zip(images, depth_maps):
        local_map = create_point_cloud(image, depth, fx, fy, cx, cy)
        global_map += local_map

    global_map = global_map.voxel_down_sample(voxel_size=0.02)
    return global_map