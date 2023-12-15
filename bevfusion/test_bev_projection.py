import cv2
import numpy as np


def read_calib_file(filepath):
    with open(filepath, 'r') as f:
        lines = f.readlines()
    return lines


def get_homography(calib_file):
    calib_data = read_calib_file(calib_file)

    # Extract necessary calibration matrices
    P_rect = np.array(calib_data[2].strip().split()[1:]).reshape(3, 4).astype(float)
    R_rect = np.eye(4)
    R_rect[:3, :3] = np.array(calib_data[4].strip().split()[1:]).reshape(3, 3).astype(float)
    Tr_velo_to_cam = np.eye(4)
    Tr_velo_to_cam[:3, :] = np.array(calib_data[5].strip().split()[1:]).reshape(3, 4).astype(float)

    # Transformation matrix from LiDAR to camera
    T_cam_to_velo = np.linalg.inv(Tr_velo_to_cam)

    # Projection matrix from camera to image coordinates
    P_cam = np.dot(P_rect, np.dot(R_rect, T_cam_to_velo))

    # Homography matrix (to transform points from camera to XY plane of LiDAR)
    H_cam_to_lidar_xy = np.array([[1, 0, 0],
                                  [0, 0, 1],
                                  [0, -1,
                                   0]])  # Assuming LiDAR XY plane is the XZ plane in the LiDAR's coordinate system

    # Calculate homography matrix between camera and LiDAR XY plane
    H_cam_to_lidar = np.dot(H_cam_to_lidar_xy, P_cam[:, :3])

    return H_cam_to_lidar

def read_camera_calib(filepath):
    with open(filepath, 'r') as f:
        lines = f.readlines()
    P2 = np.array([float(x) for x in lines[2].split()[1:13]]).reshape(3, 4)
    K = P2[:3, :3]  # Camera matrix
    return K


# Function to read image and apply homography
def create_bev_image(img, homography_matrix):
    # Apply homography to warp the image
    img_bev = cv2.warpPerspective(img, homography_matrix, (img.shape[1], img.shape[0]))

    return img_bev

# Path to KITTI calibration file for the specific sequence (change this path to your calibration file)
calib_file_path = '/home/lacie/Datasets/KITTI/objects/train/calib/000100.txt'

# Calculate the homography matrix
homography_matrix = get_homography(calib_file_path)

print("Homography Matrix between Camera and LiDAR XY plane:")
print(homography_matrix)

# Path to the image you want to use for creating BEV (change this path to your image)
image_path = '/home/lacie/Datasets/KITTI/objects/train/image_2/000100.png'

img = cv2.imread(image_path)

K = read_camera_calib(calib_file_path)
D = np.array([-3.691481e-01, 1.968681e-01, 1.353473e-03, 5.677587e-04, -6.770705e-02], dtype=np.float32)

img_undistorted = cv2.undistort(img, K, D)

H_cam_to_lidar_xy = np.array([[1, 0, 0],
                              [0, 0, 1],
                              [0, -1, 0]], dtype=np.float32)  # Assuming LiDAR XY plane is the XZ plane in the LiDAR's coordinate system

# Call the function to create BEV image
bird_eye_view_image = create_bev_image(img_undistorted, H_cam_to_lidar_xy)

# Display the original image and the BEV image
cv2.imshow('Original Image', img)
cv2.imshow('BEV Image', bird_eye_view_image)
cv2.waitKey(0)
cv2.destroyAllWindows()
