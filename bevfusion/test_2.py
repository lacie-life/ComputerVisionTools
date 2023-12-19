import cv2
import numpy as np

import os
import fnmatch
from tqdm import tqdm
from pprint import pprint
from tools import *

image_path = '/home/lacie/Datasets/KITTI/objects/train/image_2/000360.png'
pointcloud_path = '/home/lacie/Datasets/KITTI/objects/train/velodyne/000360.bin'
calib_path = '/home/lacie/Datasets/KITTI/objects/train/calib/000360.txt'


cam2cam_file = '/home/lacie/Datasets/KITTI/objects/simpleKITTI/training/global_calib/calib_cam_to_cam.txt'
velo2cam_file = '/home/lacie/Datasets/KITTI/objects/simpleKITTI/training/global_calib/calib_velo_to_cam.txt'

output_dir = './images-3/'

count = 0

def convert_point2bev(bev_image, point):
    x_index = min(max(-int(point[0] * 10) + 799, 0), bev_image.shape[0] - 1)
    y_index = min(max(int(-point[1] * 10) + 350, 0), bev_image.shape[1] - 1)
    pc_bev[x_index, y_index] = [0, 255, 0]
    print(x_index, y_index)
    # Hightlight the corner points
    cv2.rectangle(bev_image, (y_index - 5, x_index - 5), (y_index + 5, x_index + 5), (0, 255, 0), -1)  # Use green color to highlight
    cv2.imshow('pc_bev', bev_image)

def save_pixel(event, x, y, flags, param):
    global count
    if event == cv2.EVENT_LBUTTONDOWN:
        with open('clicked_pixels.txt', 'a') as f:
            rgb = pcimg[y, x]
            f.write(f'{count}: {x}, {y}, RGB: {rgb}\n')
            count += 1
        print(f'Saved pixel ({x}, {y}), RGB: {rgb}')
        cv2.circle(pcimg, (x, y), radius=0, color=(0, 0, 255), thickness=-1)  # BGR color
        cv2.imshow('pc_projected', pcimg)  # Update the image display
        convert_point2bev(pc_bev, [x, y])


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
        points[:, 1] <= maxY))
points_new = points[mask]

# points_new[:, 2] = points[:, 2]

points_new = points_new[:, :3]

corners = np.float32([[400, 250, 1], [image.shape[1] - 400, 250, 1], [image.shape[1] - 150, image.shape[0] - 10, 1], [150, image.shape[0] - 10, 1]])

vis_img = image.copy()
# draw to image
for corner in corners:
    x, y = int(corner[0]), int(corner[1])
    cv2.rectangle(vis_img, (x - 5, y - 5), (x + 5, y + 5), (0, 255, 0), -1)  # Use green color to highlight
    cv2.putText(vis_img, str(x) + ',' + str(y), (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
cv2.imwrite(output_dir + 'image_corners.png', vis_img)
# cv2.imshow('image_corners', vis_img)
# cv2.waitKey(0)

print("Corners:")
print(corners)








































# # Project lidar to image
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

source_points = np.array([[400, 249],
                            [529, 246],
                            [681, 242],
                            [844, 249],
                            [924, 297],
                            [1006, 329],
                            [1089, 364],
                            [984, 362],
                            [860, 360],
                            [748, 355],
                            [636, 353],
                            [519, 349],
                            [405, 348],
                            [288, 352],
                            [150, 360],
                            [233, 328],
                            [321, 287]])

pc_color, pc_corners = generate_colorpc(image, points_new, points_image, source_points)

print("Corner points point cloud:")
print(pc_corners)
print("pc_color:")
print(pc_color)

pc_bev = np.zeros((800, 700, 3))
for i in pc_color:
    x_index = min(max(-int(i[0] * 10) + 799, 0), pc_bev.shape[0] - 1)
    y_index = min(max(int(-i[1] * 10) + 350, 0), pc_bev.shape[1] - 1)
    pc_bev[x_index, y_index] = [i[5], i[4], i[3]]

destination_points = []

for point in pc_corners:
    x_index = min(max(-int(point[0] * 10) + 799, 0), pc_bev.shape[0] - 1)
    y_index = min(max(int(-point[1] * 10) + 350, 0), pc_bev.shape[1] - 1)
    pc_bev[x_index, y_index] = [0, 255, 0]
    print(x_index, y_index)
    # Hightlight the corner points
    # cv2.rectangle(pc_bev, (y_index - 2, x_index - 2), (y_index + 2, x_index + 2), (0, 255, 0), -1)
    destination_points.append([y_index, x_index])

cv2.imwrite(output_dir + 'pointcloud_bev.png', pc_bev)
cv2.imshow('pc_bev', pc_bev)
cv2.waitKey(0)

print(len(destination_points))
print(destination_points)
print(len(source_points))
print(source_points)

# Calculate the homography matrix
H, status = cv2.findHomography(np.array(source_points), np.array(destination_points))

# Warp source image to destination based on homography
im_out = cv2.warpPerspective(image, H, (pc_bev.shape[1], pc_bev.shape[0]))
cv2.imwrite(output_dir + 'warped_source_image.png', im_out)
cv2.imshow("Warped Source Image", im_out)
cv2.waitKey(0)































# Generate PC with Clor & Save
# pc_color, pc_corners = generate_colorpc(image, points_new, points_image, corners)
#
# print("Corner points point cloud:")
# print(pc_corners)
# print("pc_color:")
# print(pc_color)
#
# pc_bev = np.zeros((800, 700, 3))
# for i in pc_color:
#     x_index = min(max(-int(i[0] * 10) + 799, 0), pc_bev.shape[0] - 1)
#     y_index = min(max(int(-i[1] * 10) + 350, 0), pc_bev.shape[1] - 1)
#     pc_bev[x_index, y_index] = [i[5], i[4], i[3]]
    # img_bev[-int(i[0] * 10) + 799, int(-i[1] * 10) + 350] = [i[5], i[4], i[3]]

# Draw pc_corners into img_bev
# bev_corners = []
# for corner in pc_corners:
#     x_index = min(max(-int(corner[0] * 10) + 799, 0), pc_bev.shape[0] - 1)
#     y_index = min(max(int(-corner[1] * 10) + 350, 0), pc_bev.shape[1] - 1)
#     pc_bev[x_index, y_index] = [0, 255, 0]
#     print(x_index, y_index)
#     bev_corners.append([x_index, y_index])
#
# cv2.imwrite(output_dir + 'pointcloud_bev.png', pc_bev)
# cv2.imshow('pc_bev', pc_bev)
# cv2.waitKey(0)







