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

voxel_pcds = []

colors = [[0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]

index = 0

bv_images = []

for mask in [mask_below, mask_above]:

    # Translate points to positive space
    point_masked = points_new[mask]

    bv_image = make_BVFeature(point_masked)

    img = np.transpose(bv_image, (1, 2, 0))
    cv2.imwrite(output_dir + str(index) + '.png', img)
    cv2.imshow('image', img)
    cv2.waitKey(0)

    point_masked[:, :3] -= [minX, minY, minZ]

    # Convert to voxel indices
    voxel_indices = np.floor(point_masked[:, :3] / voxel_size).astype(np.int32)

    voxel_indices = np.clip(voxel_indices, a_min=0, a_max=np.array([int((maxX-minX)/voxel_size)-1, int((maxY-minY)/voxel_size)-1, int((maxZ-minZ)/voxel_size)-1]))

    # Remove duplicates
    voxel_indices = np.unique(voxel_indices, axis=0)

    voxel_pcd = o3d.geometry.PointCloud()

    voxel_center = voxel_indices * voxel_size + voxel_size / 2 + [minX, minY, minZ]
    voxel_pcd.points = o3d.utility.Vector3dVector(voxel_center)
    voxel_pcd.paint_uniform_color(colors[index])
    index += 1

    print(voxel_pcd.points)

    voxel_pcds.append(voxel_pcd)
    bv_images.append(bv_image)

end = time()

# points_new += [minX, minY, minZ]

pcd = o3d.geometry.PointCloud()
pcd.points = o3d.utility.Vector3dVector(points_new[:, :3])
pcd.paint_uniform_color([1.0, 0.0, 0.0])

voxel_pcds.append(pcd)

o3d.visualization.draw_geometries(voxel_pcds)

print('Time: ', end-start)







