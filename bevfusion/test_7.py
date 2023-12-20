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

output_dir = './images-4/'

image = cv2.imread(image_path)

# Load calibration data
P2 = read_calib_file(calib_path)

K = P2.reshape(3, 4)[:, :3]
t = P2[:, 3]

D = np.array([-3.691481e-01, 1.968681e-01, 1.353473e-03, 5.677587e-04, -6.770705e-02], dtype=np.float32)

R = np.array([[1, 0, 0], [0, 0, -1], [0, 1, 0]])

# Undistort the image
image_undistorted = cv2.undistort(image, K, D)

H = np.dot(np.dot(K, R), np.linalg.inv(K))

print("P2")
print(P2)
print("K")
print(K)
print("t")
print(t)
print("R")
print(R)
print("H")
print(H)


bev_image = cv2.warpPerspective(image_undistorted, H, (608, 608))
in_bev_image = cv2.warpPerspective(bev_image, np.linalg.inv(H), (1242, 375))

cv2.imshow('image', image_undistorted)
cv2.imshow('bev_image', bev_image)
cv2.imshow('in_bev_image', in_bev_image)
cv2.waitKey(0)

