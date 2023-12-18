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

# Remove points that are either outside or behind the camera
# Boundary condition
minX = boundary['minX']
maxX = boundary['maxX']
minY = boundary['minY']
maxY = boundary['maxY']
minZ = boundary['minZ']
maxZ = boundary['maxZ']

# Remove the point out of range x,y,z
mask = np.where((points[:, 0] >= minX) & (points[:, 0] <= maxX) & (points[:, 1] >= minY) & (
        points[:, 1] <= maxY) & (points[:, 2] >= minZ) & (points[:, 2] <= maxZ))
points_new = points[mask]
#
# points_new[:, 2] = points_new[:, 2]

points_new = points_new[:, :3]

corners = np.float32([[400, 150, 1], [image.shape[1] - 400, 150, 1], [image.shape[1] - 400, image.shape[0] - 20, 1], [400, image.shape[0] - 20, 1]])

vis_img = image.copy()
# draw to image
for corner in corners:
    x, y = int(corner[0]), int(corner[1])
    cv2.rectangle(vis_img, (x - 5, y - 5), (x + 5, y + 5), (0, 255, 0), -1)  # Use green color to highlight
cv2.imwrite(output_dir + 'image_corners.png', vis_img)

print("Corners:")
print(corners)

# Project lidar to image
points_image = project_lidar2img(image, points_new, p_matrix)

pcimg = image.copy()
depth_max = np.max(points_new[:, 0])

print("Projected points:")
print(points_image.shape)
print(points_image)

corner_points = []
bev_area = []

for idx, i in enumerate(points_image):
    color = int((points_new[idx, 0] / depth_max) * 255)
    cv2.rectangle(pcimg, (int(i[0] - 1), int(i[1] - 1)), (int(i[0] + 1), int(i[1] + 1)), (0, 0, color), -1)

    # # check i is in the corners
    # if (i[0] > 400 and i[0] < image.shape[1] - 400) and (i[1] > 150 and i[1] < image.shape[0] - 20):
    #     cv2.rectangle(pcimg, (int(i[0] - 1), int(i[1] - 1)), (int(i[0] + 1), int(i[1] + 1)), (255, 255, 0), -1)

# Find corner points in points_image
for corner in corners:
    for idx, i in enumerate(points_image):
        if (i[0] > corner[0] - 5 and i[0] < corner[0] + 5) and (i[1] > corner[1] - 5 and i[1] < corner[1] + 5):
            corner_points.append(i)
            bev_area.append(points_new[idx])

print("BEV area:")
print(len(bev_area))
print("Corner points 1:")
print(len(corner_points))
print(corner_points)

# Draw corner points
for corner in corner_points:
    cv2.rectangle(pcimg, (int(corner[0] - 1), int(corner[1] - 1)), (int(corner[0] + 1), int(corner[1] + 1)), (0, 255, 0), -1)

cv2.imwrite(output_dir + 'pointcloud_projected.png', pcimg)

# Generate PC with Clor & Save
pc_color, pc_corners = generate_colorpc(image, points_new, points_image, corners)

print("Corner points point cloud:")
print(pc_corners)
print("pc_color:")
print(pc_color)

img_bev = np.zeros((800, 700, 3))
for i in pc_color:
    x_index = min(max(-int(i[0] * 10) + 799, 0), img_bev.shape[0] - 1)
    y_index = min(max(int(-i[1] * 10) + 350, 0), img_bev.shape[1] - 1)
    img_bev[x_index, y_index] = [i[5], i[4], i[3]]
    # img_bev[-int(i[0] * 10) + 799, int(-i[1] * 10) + 350] = [i[5], i[4], i[3]]

# Draw pc_corners into img_bev
bev_corners = []
for corner in pc_corners:
    x_index = min(max(-int(corner[0] * 10) + 799, 0), img_bev.shape[0] - 1)
    y_index = min(max(int(-corner[1] * 10) + 350, 0), img_bev.shape[1] - 1)
    img_bev[x_index, y_index] = [0, 255, 0]
    print(x_index, y_index)
    cv2.putText(img_bev, str(x_index) + ',' + str(y_index), (y_index, x_index), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
    bev_corners.append([x_index, y_index])

cv2.imwrite(output_dir + 'pointcloud_bev.png', img_bev)

bev_corners = np.array(bev_corners)
original_corners = np.array([[400, 150], [image.shape[1] - 400, 150], [image.shape[1] - 400, image.shape[0] - 20], [400, image.shape[0] - 20]])


# Calculate the homography matrix from image to BEV
print("Original corners:")
print(original_corners)
print("BEV corners:")
print(bev_corners)


H, status = cv2.findHomography(np.float32(original_corners), np.float32(bev_corners))
print("Homography matrix:")
print(H)

# Warp image to BEV
img_bev = cv2.warpPerspective(image, H, (img_bev.shape[1], img_bev.shape[0]))
cv2.imwrite(output_dir + 'image_bev.png', img_bev)





