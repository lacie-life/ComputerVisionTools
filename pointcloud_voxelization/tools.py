import numpy as np
import cv2
from pprint import pprint

boundary = {
    "minX": 0,
    "maxX": 50,
    "minY": -25,
    "maxY": 25,
    "minZ": -2.73,
    "maxZ": 1.27
}

DISCRETIZATION = (boundary["maxX"] - boundary["minX"]) / 608

def load_calib_cam2cam(filename, debug=False):
    """
    Only load R_rect & P_rect for neeed
    Parameters: filename of the calib file
    Return:
        R_rect: a list of r_rect(shape:3*3)
        P_rect: a list of p_rect(shape:3*4)
    """
    with open(filename) as f_calib:
        lines = f_calib.readlines()

    R_rect = []
    P_rect = []

    for line in lines:
        title = line.strip().split(' ')[0]
        if title[:-4] == "R_rect":
            r_r = np.array(line.strip().split(' ')[1:], dtype=np.float32)
            r_r = np.reshape(r_r, (3, 3))
            R_rect.append(r_r)
        elif title[:-4] == "P_rect":
            p_r = np.array(line.strip().split(' ')[1:], dtype=np.float32)
            p_r = np.reshape(p_r, (3, 4))
            P_rect.append(p_r)

    if debug:
        print("R_rect:")
        pprint(R_rect)

        print()
        print("P_rect:")
        pprint(P_rect)

    return R_rect, P_rect


def load_calib_lidar2cam(filename, debug=False):
    """
    Load calib
    Parameters: filename of the calib file
    Return:
        tr: shape(4*4)
            [  r   t
             0 0 0 1]
    """
    with open(filename) as f_calib:
        lines = f_calib.readlines()

    for line in lines:
        title = line.strip().split(' ')[0]
        if title[:-1] == "R":
            r = np.array(line.strip().split(' ')[1:], dtype=np.float32)
            r = np.reshape(r, (3, 3))
        if title[:-1] == "T":
            t = np.array(line.strip().split(' ')[1:], dtype=np.float32)
            t = np.reshape(t, (3, 1))

    tr = np.hstack([r, t])
    tr = np.vstack([tr, np.array([0, 0, 0, 1])])

    if debug:
        print()
        print("Tr:")
        print(tr)

    return tr


def load_calib(filename, debug=False):
    """
    Load the calib parameters which has R_rect & P_rect & Tr in the same file
    Parameters:
        filename: the filename of the calib file
    Return:
        R_rect, P_rect, Tr
    """
    with open(filename) as f_calib:
        lines = f_calib.readlines()

        P_rect = []
    for line in lines:
        title = line.strip().split(' ')[0]
        if len(title):
            if title[0] == "R":
                R_rect = np.array(line.strip().split(' ')[1:], dtype=np.float32)
                R_rect = np.reshape(R_rect, (3, 3))
            elif title[0] == "P":
                p_r = np.array(line.strip().split(' ')[1:], dtype=np.float32)
                p_r = np.reshape(p_r, (3, 4))
                P_rect.append(p_r)
            elif title[:-1] == "Tr_velo_to_cam":
                Tr = np.array(line.strip().split(' ')[1:], dtype=np.float32)
                Tr = np.reshape(Tr, (3, 4))
                Tr = np.vstack([Tr, np.array([0, 0, 0, 1])])

    return R_rect, P_rect, Tr


def cal_proj_matrix_raw(filename_c2c, filename_l2c, camera_id, debug=False):
    """
    Compute the projection matrix from LiDAR to Img
    Parameters:
        filename_c2c: filename of the calib file for cam2cam
        filename_l2c: filename of the calib file for lidar2cam
        camera_id: the NO. of camera
    Return:
        P_lidar2img: the projection matrix from LiDAR to Img
    """
    # Load Calib Parameters
    R_rect, P_rect = load_calib_cam2cam(filename_c2c, debug)
    tr = load_calib_lidar2cam(filename_l2c, debug)

    # Calculation
    R_cam2rect = np.hstack([R_rect[0], np.array([[0], [0], [0]])])
    R_cam2rect = np.vstack([R_cam2rect, np.array([0, 0, 0, 1])])

    P_lidar2img = np.matmul(P_rect[camera_id], R_cam2rect)
    P_lidar2img = np.matmul(P_lidar2img, tr)

    if debug:
        print()
        print("P_lidar2img:")
        print(P_lidar2img)

    return P_lidar2img


def cal_proj_matrix(filename, camera_id, debug=False):
    """
    Compute the projection matrix from LiDAR to Img
    Parameters:
        filename: filename of the calib file
        camera_id: the NO. of camera
    Return:
        P_lidar2img: the projection matrix from LiDAR to Img
    """
    # Load Calib Parameters
    R_rect, P_rect, tr = load_calib(filename, debug)

    # Calculation
    R_cam2rect = np.hstack([R_rect, np.array([[0], [0], [0]])])
    R_cam2rect = np.vstack([R_cam2rect, np.array([0, 0, 0, 1])])

    P_lidar2img = np.matmul(P_rect[camera_id], R_cam2rect)
    P_lidar2img = np.matmul(P_lidar2img, tr)

    if debug:
        print()
        print("P_lidar2img:")
        print(P_lidar2img)

    return P_lidar2img


def project_lidar2img(img, pc, p_matrix, debug=False):
    """
    Project the LiDAR PointCloud to Image
    Parameters:
        img: Image
        pc: PointCloud
        p_matrix: projection matrix
    """
    # Dimension of data & projection matrix
    dim_norm = p_matrix.shape[0]
    dim_proj = p_matrix.shape[1]

    # Do transformation in homogenuous coordinates
    pc_temp = pc.copy()
    if pc_temp.shape[1] < dim_proj:
        pc_temp = np.hstack([pc_temp, np.ones((pc_temp.shape[0], 1))])
    points = np.matmul(p_matrix, pc_temp.T)
    points = points.T

    temp = np.reshape(points[:, dim_norm - 1], (-1, 1))
    points = points[:, :dim_norm] / (np.matmul(temp, np.ones([1, dim_norm])))

    return points


def generate_colorpc(img, pc, pcimg, sample_points=[], debug=False):
    """
    Generate the PointCloud with color
    Parameters:
        img: image
        pc: PointCloud
        pcimg: PointCloud project to image
    Return:
        pc_color: PointCloud with color e.g. X Y Z R G B
    """
    x = np.reshape(pcimg[:, 0], (-1, 1))
    y = np.reshape(pcimg[:, 1], (-1, 1))
    xy = np.hstack([x, y])

    pc_color = []
    for idx, i in enumerate(xy):
        if (i[0] > 1 and i[0] < img.shape[1]) and (i[1] > 1 and i[1] < img.shape[0]):
            bgr = img[int(i[1]), int(i[0])]
            p_color = [pc[idx][0], pc[idx][1], pc[idx][2], bgr[2], bgr[1], bgr[0]]
            pc_color.append(p_color)

            # Draw corner area
            # for corner in sample_points:
            #     cv2.rectangle(img, (int(corner[0] - 1), int(corner[1] - 1)), (int(corner[0] + 1), int(corner[1] + 1)), (0, 255, 0), -1)

    if len(sample_points) != 0:
        # Find 4 point closest to the corner
        print("Corner points 2:")
        print(sample_points)
        pc_corners = []
        for corner in sample_points:
            min_dist = 100000
            min_point = []
            for idx, i in enumerate(xy):
                dist = np.sqrt((i[0] - corner[0]) ** 2 + (i[1] - corner[1]) ** 2)
                if dist < min_dist:
                    min_dist = dist
                    min_point = pc[idx]
            pc_corners.append(min_point)

        pc_color = np.array(pc_color)
        pc_corners = np.array(pc_corners)

        return pc_color, pc_corners
    return pc_color

def read_calib_file(filepath):
    data = {}
    with open(filepath, 'r') as file:
        for line in file.readlines():
            if 'P2' in line:
                calib_values = line.strip().split(' ')
                camera_matrix = np.array([float(val) for val in calib_values[1:]]).reshape(3, 4)
    return camera_matrix

def make_BVFeature(point_cloud, Discretization=DISCRETIZATION, bc=boundary):
    Height = 608 + 1
    Width = 608 + 1

    print("point_cloud.shape:", point_cloud.shape)

    # Discretize Feature Map
    PointCloud = np.copy(point_cloud)
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



