import cv2
import numpy as np

# Load image and calibration data
image = cv2.imread('/home/lacie/Datasets/KITTI/objects/train/image_2/000000.png')
calib_file = '/home/lacie/Datasets/KITTI/objects/train/calib/000000.txt'

def read_calib_file(filepath):

    data = {}
    with open(filepath, 'r') as file:
        for line in file.readlines():
            if 'P2' in line:
                calib_values = line.strip().split(' ')
                camera_matrix = np.array([float(val) for val in calib_values[1:]]).reshape(3, 4)
    return camera_matrix

P2 = read_calib_file(calib_file)

K = P2.reshape(3, 4)[:, :3]
D = np.array([-3.691481e-01, 1.968681e-01, 1.353473e-03, 5.677587e-04, -6.770705e-02], dtype=np.float32)

print(P2)

sz = (608, 608)

image_undistorted = cv2.undistort(image, K, D)

cv2.imshow('image', image)
cv2.waitKey(0)

# Define source and destination points for the perspective transformation
src = np.float32([[0, image.shape[0]], [image.shape[1], image.shape[0]], [0, 0], [image.shape[1], 0]])
dst = np.float32([[0, sz[1]], [sz[0], sz[1]], [0, 0], [sz[0], 0]])

# Compute the perspective transform, M
M = cv2.getPerspectiveTransform(src, dst)

# Apply perspective transformation to get bird's eye view
birdseye_view = cv2.warpPerspective(image_undistorted, M, sz)

# Display or save the Bird's Eye View image
cv2.imshow('BEV Image', birdseye_view)
cv2.waitKey(0)
cv2.destroyAllWindows()


