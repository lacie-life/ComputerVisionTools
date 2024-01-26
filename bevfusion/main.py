import cv2
import numpy as np
import time

# Load image and calibration data
image = cv2.imread('/home/lacie/Datasets/KITTI/objects/train/image_2/000100.png')
point_cloud = np.fromfile('/home/lacie/Datasets/KITTI/objects/train/velodyne/000100.bin', dtype=np.float32).reshape(-1,
                                                                                                                    4)
calib_file = '/home/lacie/Datasets/KITTI/objects/train/calib/000100.txt'
cam2cam_file = '/home/lacie/Datasets/KITTI/objects/simpleKITTI/training/global_calib/calib_cam_to_cam.txt'
velo2cam_file = '/home/lacie/Datasets/KITTI/objects/simpleKITTI/training/global_calib/calib_velo_to_cam.txt'

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

def read_cam2lidar_calib_file(filepath):
    data = {}
    with open(filepath, 'r') as file:
        for line in file.readlines():
            if 'R0_rect' in line:
                calib_values = line.strip().split(' ')
                R0_rect = np.array([float(val) for val in calib_values[1:]]).reshape(3, 3)
            if 'Tr_velo_to_cam' in line:
                calib_values = line.strip().split(' ')
                Tr_velo_to_cam = np.array([float(val) for val in calib_values[1:]]).reshape(3, 4)

    T_cam_lidar = np.linalg.inv(np.vstack([Tr_velo_to_cam, [0, 0, 0, 1]]))

    return R0_rect, Tr_velo_to_cam, T_cam_lidar


def make_BVFeature(point_cloud, Discretization=DISCRETIZATION, bc=boundary):
    Height = 608 + 1
    Width = 608 + 1

    # Boundary condition
    minX = boundary['minX']
    maxX = boundary['maxX']
    minY = boundary['minY']
    maxY = boundary['maxY']
    minZ = boundary['minZ']
    maxZ = boundary['maxZ']

    # Remove the point out of range x,y,z
    mask = np.where((point_cloud[:, 0] >= minX) & (point_cloud[:, 0] <= maxX) & (point_cloud[:, 1] >= minY) & (
            point_cloud[:, 1] <= maxY) & (point_cloud[:, 2] >= minZ) & (point_cloud[:, 2] <= maxZ))
    PointCloud = point_cloud[mask]

    PointCloud[:, 2] = PointCloud[:, 2] - minZ

    # Discretize Feature Map
    PointCloud = np.copy(PointCloud)
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

def project_image_to_lidar_bev (image, calib_file):

    minX = boundary['minX']
    maxX = boundary['maxX']

    image_discretization = (maxX - minX) / 1224

    # Load calibration data
    R0_rect, Tr_velo_to_cam, T_cam_lidar = read_cam2lidar_calib_file(calib_file)

    # Project image point to 3D points in camera coordinate
    height, width, _ = image.shape
    x = np.linspace(0, width - 1, width)
    y = np.linspace(0, height - 1, height)
    x, y = np.meshgrid(x, y)
    homogenous_image_points = np.stack([x, y, np.ones_like(x)], axis=2).reshape(-1, 3)

    camera_points = np.linalg.inv(P2[:, :3]) @ homogenous_image_points.T

    print("Camera points")
    print(camera_points)

    camera_points_homogenous = np.vstack([camera_points, np.ones(camera_points.shape[1])])

    # Project 3D points in camera coordinate to 3D points in lidar coordinate
    lidar_points = T_cam_lidar @ camera_points_homogenous

    print("Lidar points")
    print(lidar_points)

    print(np.max(lidar_points[0, :]))
    print(np.min(lidar_points[0, :]))

    ratio = np.max(lidar_points[0, :]) / np.min(lidar_points[0, :]) / 1224

    # Project 3D points in lidar coordinate to 2D points in lidar BEV
    bev_points = lidar_points[:2, :]
    bev_points[0, :] = np.int_(np.floor(bev_points[0, :] / image_discretization)).astype(int)
    bev_points[1, :] = np.int_(np.floor(bev_points[1, :] / image_discretization) + 608 / 2).astype(int)

    print("BEV points")
    print(bev_points)

    finite_indices = np.isfinite(bev_points).all(axis=0)
    bev_points = bev_points[:, finite_indices]

    bev_points = np.int_(bev_points)

    bev_points[0, bev_points[0, :] >= 608] = 607
    bev_points[1, bev_points[1, :] >= 608] = 607

    bev_image = np.zeros((608, 608, 3), dtype=np.uint8)

    for i in range(min(bev_points.shape[1], y.shape[1])):
        x_coord = int(y[0, i])
        y_coord = int(x[0, i])
        # print(x_coord, y_coord)
        # print(bev_points[0, i], bev_points[1, i])
        if x_coord < 608 and y_coord < 608:
            rgb_value = image[x_coord, y_coord, :]
            bev_image[bev_points[0, i], bev_points[1, i], :] = rgb_value

    print(bev_image)

    return bev_image


P2 = read_calib_file(calib_file)

K = P2.reshape(3, 4)[:, :3]
D = np.array([-3.691481e-01, 1.968681e-01, 1.353473e-03, 5.677587e-04, -6.770705e-02], dtype=np.float32)

print(P2)

sz = (608, 608)

image_undistorted = cv2.undistort(image, K, D)

# cv2.imshow('image undistorted', image_undistorted)
# cv2.waitKey(0)

# image_bev = project_image_to_lidar_bev(image_undistorted, calib_file)
t1 = time.time()
lidar_bev = make_BVFeature(point_cloud)
t2 = time.time()

print("Time: ", t2 - t1)
# cv2.imwrite('camera_birdseye_view.png', image_bev)
# # Display or save the Bird's Eye View image
# cv2.imshow('BEV Image', image_bev)
# cv2.waitKey(0)

print(lidar_bev.shape)

lidar_bev_transposed = np.transpose(lidar_bev, (1, 2, 0))

# Convert the data type to uint8
lidar_bev_transposed = (lidar_bev_transposed * 255).astype(np.uint8)

cv2.imwrite('lidar_birdseye_view.png', lidar_bev_transposed)
# Display or save the Bird's Eye View image
cv2.imshow('BEV Image', lidar_bev_transposed)

cv2.waitKey(0)
cv2.destroyAllWindows()
