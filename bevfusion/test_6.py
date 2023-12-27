
import os
import cv2
import numpy as np
from tools import *

# Specify the directory you want to read from
image_directory = '/home/lacie/Datasets/KITTI/objects/train/image_2/'
calib_directory = '/home/lacie/Datasets/KITTI/objects/train/calib/'
pointcloud_path = '/home/lacie/Datasets/KITTI/objects/train/velodyne/'


cam2cam_file = '/home/lacie/Datasets/KITTI/objects/simpleKITTI/training/global_calib/calib_cam_to_cam.txt'
velo2cam_file = '/home/lacie/Datasets/KITTI/objects/simpleKITTI/training/global_calib/calib_velo_to_cam.txt'

# Get a list of all files in the directory
image_files = os.listdir(image_directory)
#
# H = np.array([-0.099134, -1.947306, 402.704874,
#               -0.026298, -4.464355, 868.110926,
#               -0.000025, -0.005596, 1.000000]).reshape(3, 3)

# H = np.array([-0.097938, -1.684271, 355.704874,
#               -0.021413, -3.389828, 676.110926,
#               -0.000025, -0.005596, 1.000000]).reshape(3, 3)

# H = np.array([-0.131077, -1.650271, 360.800331,
#               -0.069342, -3.331980, 682.156221,
#               -0.000120, -0.005446, 1.000000]).reshape(3, 3)

H = np.array([-0.171769, -1.602777, 368.739276,
              -0.143001, -3.195313, 686.580487,
              -0.000250, -0.005173, 1.000000]).reshape(3, 3)

boundary = {
    "minX": 0,
    "maxX": 50,
    "minY": -25,
    "maxY": 25,
    "minZ": -2.73,
    "maxZ": 1.27
}

# Remove points that are either outside or behind the camera
# Boundary condition
minX = boundary['minX']
maxX = boundary['maxX']
minY = boundary['minY']
maxY = boundary['maxY']
minZ = boundary['minZ']
maxZ = boundary['maxZ']

p_matrix = cal_proj_matrix_raw(cam2cam_file, velo2cam_file, 2)

# Iterate over each file
for image_file in image_files:
    # Check if the file is an image
    if image_file.endswith('.png') or image_file.endswith('.jpg') or image_file.endswith('.jpeg'):
        # Construct the full file path
        image_path = os.path.join(image_directory, image_file)
        calib_path = os.path.join(calib_directory, image_file.replace('.png', '.txt'))
        pc_path = os.path.join(pointcloud_path, image_file.replace('.png', '.bin'))

        # Load image and calibration data
        image = cv2.imread(image_path)
        points = np.fromfile(pc_path, dtype=np.float32).reshape(-1, 4)
        calib_file = calib_path

        # Remove the point out of range x,y,z
        mask = np.where((points[:, 0] >= minX) & (points[:, 0] <= maxX) & (points[:, 1] >= minY) & (
                points[:, 1] <= maxY))
        points_new = points[mask]

        # points_new[:, 2] = points[:, 2]

        points_new = points_new[:, :3]

        # Load calibration data
        P2 = read_calib_file(calib_file)
        K = P2.reshape(3, 4)[:, :3]
        D = np.array([-3.691481e-01, 1.968681e-01, 1.353473e-03, 5.677587e-04, -6.770705e-02], dtype=np.float32)

        # Undistort the image
        image_undistorted = cv2.undistort(image, K, D)

        # Project point cloud to image
        points_image = project_lidar2img(image_undistorted, points_new, p_matrix)

        pcimg = image_undistorted.copy()
        depth_max = np.max(points_new[:, 0])

        corner_points = []
        bev_area = []

        for idx, i in enumerate(points_image):
            color = int((points_new[idx, 0] / depth_max) * 255)
            cv2.rectangle(pcimg, (int(i[0] - 1), int(i[1] - 1)), (int(i[0] + 1), int(i[1] + 1)), (0, 0, color), -1)

        # Project color to point cloud
        pc_color = generate_colorpc(image_undistorted, points_new, points_image)

        # Create pc_bev
        pc_bev = np.zeros((608, 608, 3))
        for i in pc_color:
            x_index = min(max(-int(i[0] * 10) + 607, 0), pc_bev.shape[0] - 1)
            y_index = min(max(int(-i[1] * 10) + 303, 0), pc_bev.shape[1] - 1)
            pc_bev[x_index, y_index] = [i[5], i[4], i[3]]

        # # Project image to lidar BEV
        bev_image = cv2.warpPerspective(image_undistorted, H, (608, 608))
        inv_bev_image = cv2.warpPerspective(bev_image, np.linalg.inv(H), (1242, 375))
        #
        # # Save the BEV image
        # cv2.imwrite(f'bev_{image_file}', image_bev)

        pc_bev = pc_bev.astype(np.uint8)

        merge_img = cv2.addWeighted(bev_image, 0.5, pc_bev, 0.5, 0)

        cv2.imshow('image', image)
        cv2.imshow('image_undistorted', image_undistorted)
        cv2.imshow('bev_image', bev_image)
        cv2.imshow('inv_bev_image', inv_bev_image)
        cv2.imshow('pc_bev', pc_bev.astype(np.uint8))
        cv2.imshow('merge_img', merge_img.astype(np.uint8))
        cv2.waitKey(0)


