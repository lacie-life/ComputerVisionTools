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

# Load calibration data
p_matrix = cal_proj_matrix_raw(cam2cam_file, velo2cam_file, 2)

# Load image and point cloud
image = cv2.imread(image_path)
points = np.fromfile(pointcloud_path, dtype=np.float32).reshape(-1, 4)
points = points[:, :3]
# points.tofile("./temp_pc.bin")

# Remove all points behind image plane (approximation)
# cloud = PyntCloud.from_file("./temp_pc.bin")
# cloud.points = cloud.points[cloud.points["x"] >= 0]
# points = np.array(cloud.points)

points_image = project_lidar2img(image, points, p_matrix)

pcimg = image.copy()
depth_max = np.max(points[:, 0])

for idx, i in enumerate(points_image):
    color = int((points[idx, 0] / depth_max) * 255)
    cv2.rectangle(pcimg, (int(i[0] - 1), int(i[1] - 1)), (int(i[0] + 1), int(i[1] + 1)), (0, 0, color), -1)

cv2.imwrite('pointcloud_projected.png', pcimg)

# Generate PC with Clor & Save
pc_color = generate_colorpc(image, points, points_image)

img_bev = np.zeros((800, 700, 3))
for i in pc_color:
    x_index = min(max(-int(i[0] * 10) + 799, 0), img_bev.shape[0] - 1)
    y_index = min(max(int(-i[1] * 10) + 350, 0), img_bev.shape[1] - 1)
    img_bev[x_index, y_index] = [i[5], i[4], i[3]]
    # img_bev[-int(i[0] * 10) + 799, int(-i[1] * 10) + 350] = [i[5], i[4], i[3]]

cv2.imwrite('pointcloud_bev.png', img_bev)
