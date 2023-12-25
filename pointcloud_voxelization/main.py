import numpy as np
from time import time
import open3d as o3d
import cv2
import os
from tools import *

image_path = '/home/lacie/Datasets/KITTI/objects/train/image_2/000360.png'
pointcloud_path = '/home/lacie/Datasets/KITTI/objects/train/velodyne/000360.bin'
calib_path = '/home/lacie/Datasets/KITTI/objects/train/calib/000360.txt'

cam2cam_file = '/home/lacie/Datasets/KITTI/objects/simpleKITTI/training/global_calib/calib_cam_to_cam.txt'
velo2cam_file = '/home/lacie/Datasets/KITTI/objects/simpleKITTI/training/global_calib/calib_velo_to_cam.txt'

output_dir = './images/'

if not os.path.exists(output_dir):
    os.makedirs(output_dir)

voxel_size = 0.5

boundary = {
    "minX": 0,
    "maxX": 50,
    "minY": -25,
    "maxY": 25,
    "minZ": -2.73,
    "maxZ": 1.27
}

image = cv2.imread(image_path)
points = np.fromfile(pointcloud_path, dtype=np.float32).reshape(-1, 4)


P = cal_proj_matrix_raw(cam2cam_file, velo2cam_file, 2)

# Remove points that are either outside or behind the camera
# Boundary condition
minX = boundary['minX']
maxX = boundary['maxX']
minY = boundary['minY']
maxY = boundary['maxY']
minZ = boundary['minZ']
maxZ = boundary['maxZ']
midZ = (minZ + maxZ) / 2

mask = np.where((points[:, 0] >= minX) & (points[:, 0] <= maxX) & (points[:, 1] >= minY) & (
        points[:, 1] <= maxY))
points_new = points[mask]

# points_new = points_new[:, :3]

# Project the 3D points to the 2D image plane
points_homogeneous = np.hstack((points_new[:, :3], np.ones((points_new.shape[0], 1))))
points_2d = np.dot(P, points_homogeneous.T).T
points_2d[:, :2] /= points_2d[:, 2:]

# Remove the points that fall outside the image boundaries
mask = (points_2d[:, 0] >= 0) & (points_2d[:, 0] < image.shape[1]) & (points_2d[:, 1] >= 0) & (points_2d[:, 1] < image.shape[0])
points_new = points_new[mask]

mask_below = points_new[:, 2] < midZ
mask_above = points_new[:, 2] >= midZ

start = time()

colors = [[0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]

# Define the voxel size and grid range
voxel_size = np.array([0.5, 0.5, 0.5])  # Example voxel size
grid_range = np.array([0, -25, -2.73, 50, 25, 1.27])  # Example grid range

index = 0

bv_images = []

all_voxels = []
all_coordinates = []
all_num_points_per_voxel = []

voxel_pcds = []

for i, mask in enumerate([mask_below, mask_above]):

    # Translate points to positive space
    point_masked = points_new[mask]

    bv_image = make_BVFeature(point_masked)
    #
    # img = np.transpose(bv_image, (1, 2, 0))
    # cv2.imwrite(output_dir + str(index) + '.png', img)
    # cv2.imshow('image', img)
    # cv2.waitKey(0)

    voxels, coordinates, num_points_per_voxel = voxelize(point_masked, voxel_size, grid_range, max_points_in_voxel=35, max_num_voxels=20000)

    all_voxels.append(voxels)
    all_coordinates.append(coordinates)
    all_num_points_per_voxel.append(num_points_per_voxel)

    print('voxels: ', voxels.shape)
    print('coordinates: ', coordinates.shape)
    print('num_points_per_voxel: ', num_points_per_voxel.shape)

    bv_images.append(bv_image)

end = time()

# points_new += [minX, minY, minZ]

# pcd = o3d.geometry.PointCloud()
# pcd.points = o3d.utility.Vector3dVector(points_new[:, :3])
# pcd.paint_uniform_color([1.0, 0.0, 0.0])
#
# voxel_pcds.append(pcd)
#
# o3d.visualization.draw_geometries(voxel_pcds)

print('Time: ', end-start)

all_voxel_pcd = o3d.geometry.PointCloud()

colors = [[0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]

color_index = 0

# Calculate the voxel centers
voxel_centers_1 = []
voxel_centers_2 = []
for voxels in all_voxels:
    if color_index == 0:
        for voxel in voxels:
            voxel_center = voxel[:, :3] * voxel_size + voxel_size / 2
            voxel_centers_1.append(voxel_center)
    else:
        for voxel in voxels:
            voxel_center = voxel[:, :3] * voxel_size + voxel_size / 2
            voxel_centers_2.append(voxel_center)
    color_index += 1

# Create point cloud for voxel centers 1
voxel_centers_1 = np.concatenate(voxel_centers_1, axis=0)
voxel_pcd_1 = o3d.geometry.PointCloud()
voxel_pcd_1.points = o3d.utility.Vector3dVector(voxel_centers_1)
voxel_pcd_1.paint_uniform_color(colors[0])

# Create point cloud for voxel centers 2
voxel_centers_2 = np.concatenate(voxel_centers_2, axis=0)
voxel_pcd_2 = o3d.geometry.PointCloud()
voxel_pcd_2.points = o3d.utility.Vector3dVector(voxel_centers_2)
voxel_pcd_2.paint_uniform_color(colors[1])

# Create point cloud for all voxel centers
# all_voxel_pcd.append(voxel_pcd_1)
# all_voxel_pcd.append(voxel_pcd_2)

o3d.visualization.draw_geometries([voxel_pcd_1, voxel_pcd_2])











