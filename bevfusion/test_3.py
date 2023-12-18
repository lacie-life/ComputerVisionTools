import cv2
import numpy as np

import os
import fnmatch
from tqdm import tqdm
from pprint import pprint
from tools import *

image_path = '/home/lacie/Datasets/KITTI/objects/train/image_2/000100.png'
pointcloud_path = '/home/lacie/Datasets/KITTI/objects/train/velodyne/000100.bin'
calib_path = '/home/lacie/Datasets/KITTI/objects/train/calib/000100.txt'


cam2cam_file = '/home/lacie/Datasets/KITTI/objects/simpleKITTI/training/global_calib/calib_cam_to_cam.txt'
velo2cam_file = '/home/lacie/Datasets/KITTI/objects/simpleKITTI/training/global_calib/calib_velo_to_cam.txt'

output_dir = './images/'

boundary = {
    "minX": 0,
    "maxX": 50,
    "minY": -25,
    "maxY": 25,
    "minZ": -2.73,
    "maxZ": 1.27
}

# Load calibration data
p_matrix = cal_proj_matrix_raw(cam2cam_file, velo2cam_file, 2)

# Load image and point cloud
image = cv2.imread(image_path)
points = np.fromfile(pointcloud_path, dtype=np.float32).reshape(-1, 4)

points_new = points[:, :3]

P2 = read_calib_file(calib_path)

K = P2.reshape(3, 4)[:, :3]
D = np.array([-3.691481e-01, 1.968681e-01, 1.353473e-03, 5.677587e-04, -6.770705e-02], dtype=np.float32)

# undistort image
image_undistorted = cv2.undistort(image, K, D)

cv2.imshow('image', image)
cv2.imshow('image_undistorted', image_undistorted)
# cv2.waitKey(0)

points_with_z_zero = points[points[:, 2] == 0]

# Project these points to the image
points_image_with_z_zero = project_lidar2img(image_undistorted, points_with_z_zero, p_matrix)

z_points = []
for point in points_image_with_z_zero:
    x, y = int(point[0]), int(point[1])
    cv2.circle(image_undistorted, (x, y), 2, (0, 255, 0), -1)  # Draw a green circle at each point
    if x < image_undistorted.shape[1] and x > 0 and y < image_undistorted.shape[0] and y > 0:
        z_points.append([x, y])

print("z_points:")
print(z_points)
z_points = np.array(z_points, dtype=np.int32).reshape((-1, 1, 2))

cv2.polylines(image_undistorted, [z_points], isClosed=False, color=(0, 255, 0), thickness=2)

# Display the image
cv2.imshow('Image with projected points', image_undistorted)
cv2.waitKey(0)





