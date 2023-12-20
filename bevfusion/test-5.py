
import os
import cv2
import numpy as np
from tools import *

# Specify the directory you want to read from
image_directory = '/home/lacie/Datasets/KITTI/objects/train/image_2/'
calib_directory = '/home/lacie/Datasets/KITTI/objects/train/calib/'

# Get a list of all files in the directory
image_files = os.listdir(image_directory)
#
# H = np.array([-0.099134, -1.947306, 402.704874,
#               -0.026298, -4.464355, 868.110926,
#               -0.000025, -0.005596, 1.000000]).reshape(3, 3)

H = np.array([-0.097938, -1.684271, 355.704874,
              -0.021413, -3.389828, 676.110926,
              -0.000025, -0.005596, 1.000000]).reshape(3, 3)


# Iterate over each file
for image_file in image_files:
    # Check if the file is an image
    if image_file.endswith('.png') or image_file.endswith('.jpg') or image_file.endswith('.jpeg'):
        # Construct the full file path
        image_path = os.path.join(image_directory, image_file)
        calib_path = os.path.join(calib_directory, image_file.replace('.png', '.txt'))

        # Load image and calibration data
        image = cv2.imread(image_path)
        calib_file = calib_path

        # Load calibration data
        P2 = read_calib_file(calib_file)
        K = P2.reshape(3, 4)[:, :3]
        D = np.array([-3.691481e-01, 1.968681e-01, 1.353473e-03, 5.677587e-04, -6.770705e-02], dtype=np.float32)

        # Undistort the image
        image_undistorted = cv2.undistort(image, K, D)

        # # Project image to lidar BEV
        bev_image = cv2.warpPerspective(image_undistorted, H, (608, 608))
        inv_bev_image = cv2.warpPerspective(bev_image, np.linalg.inv(H), (1242, 375))
        #
        # # Save the BEV image
        # cv2.imwrite(f'bev_{image_file}', image_bev)

        cv2.imshow('image', image)
        cv2.imshow('image_undistorted', image_undistorted)
        cv2.imshow('bev_image', bev_image)
        cv2.imshow('inv_bev_image', inv_bev_image)
        cv2.waitKey(0)


