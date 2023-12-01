import cv2
import numpy as np

# Load image and calibration data
image = cv2.imread('/home/lacie/Datasets/KITTI/objects/train/image_2/000000.png')
point_cloud = np.fromfile('/home/lacie/Datasets/KITTI/objects/train/velodyne/000000.bin', dtype=np.float32).reshape(-1, 4)
calib_file = '/home/lacie/Datasets/KITTI/objects/train/calib/000000.txt'

# Back back (of vehicle) Point Cloud boundary for BEV
boundary = {
    "minX": 0,
    "maxX": 50,
    "minY": -25,
    "maxY": 25,
    "minZ": -2.73,
    "maxZ": 1.27
}

DISCRETIZATION = (boundary["maxX"] - boundary["minX"]) / 608

def read_calib_file(filepath):

    data = {}
    with open(filepath, 'r') as file:
        for line in file.readlines():
            if 'P2' in line:
                calib_values = line.strip().split(' ')
                camera_matrix = np.array([float(val) for val in calib_values[1:]]).reshape(3, 4)
    return camera_matrix


def makeBVFeature(PointCloud_, Discretization = DISCRETIZATION, bc = boundary):
    Height = 608 + 1
    Width = 608 + 1

    # Discretize Feature Map
    PointCloud = np.copy(PointCloud_)
    PointCloud[:, 0] = np.int_(np.floor(PointCloud[:, 0] / Discretization))
    PointCloud[:, 1] = np.int_(np.floor(PointCloud[:, 1] / Discretization) + Width / 2)

    # sort-3times
    indices = np.lexsort((-PointCloud[:, 2], PointCloud[:, 1], PointCloud[:, 0]))
    PointCloud = PointCloud[indices]

    # Height Map
    heightMap = np.zeros((Height, Width))

    _, indices = np.unique(PointCloud[:, 0:2], axis=0, return_index=True)
    PointCloud_frac = PointCloud[indices]
    # some important problem is image coordinate is (y,x), not (x,y)
    max_height = float(np.abs(bc['maxZ'] - bc['minZ']))
    heightMap[np.int_(PointCloud_frac[:, 0]), np.int_(PointCloud_frac[:, 1])] = PointCloud_frac[:, 2] / max_height

    # Intensity Map & DensityMap
    intensityMap = np.zeros((Height, Width))
    densityMap = np.zeros((Height, Width))

    _, indices, counts = np.unique(PointCloud[:, 0:2], axis=0, return_index=True, return_counts=True)
    PointCloud_top = PointCloud[indices]

    normalizedCounts = np.minimum(1.0, np.log(counts + 1) / np.log(64))

    intensityMap[np.int_(PointCloud_top[:, 0]), np.int_(PointCloud_top[:, 1])] = PointCloud_top[:, 3]
    densityMap[np.int_(PointCloud_top[:, 0]), np.int_(PointCloud_top[:, 1])] = normalizedCounts

    RGB_Map = np.zeros((3, Height - 1, Width - 1))

    RGB_Map[2, :, :] = densityMap[:608, :608]  # r_map

    RGB_Map[1, :, :] = heightMap[:608, :608]  # g_map

    RGB_Map[0, :, :] = intensityMap[:608, :608]  # b_map

    return RGB_Map

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


