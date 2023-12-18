import numpy as np
import cv2

cam2cam_file = '/home/lacie/Datasets/KITTI/objects/simpleKITTI/training/global_calib/calib_cam_to_cam.txt'
velo2cam_file = '/home/lacie/Datasets/KITTI/objects/simpleKITTI/training/global_calib/calib_velo_to_cam.txt'
image_path = '/home/lacie/Datasets/KITTI/objects/train/image_2/000100.png'
calib_path = '/home/lacie/Datasets/KITTI/objects/train/calib/000100.txt'

def project_velo2cam(ground_plane, P_rect, R_rect, T_rect, velo2cam, maxHeight, maxWidth):
    # Define the maximum and minimum height of the point cloud
    min_height = -1.5
    max_height = 0.5

    # Define the number of bins in the bird's eye view
    n_bins = 512

    # Define the height of each bin
    bin_height = (max_height - min_height) / n_bins

    # Define the projection matrix
    P_velo2cam = np.hstack((R_rect, T_rect))
    P_velo2cam = np.vstack((P_velo2cam, [0, 0, 0, 1]))

    # Define the inverse projection matrix
    P_cam2velo = np.linalg.inv(P_velo2cam)

    # Define the ground plane in camera coordinates
    ground_plane_cam = np.dot(P_cam2velo, ground_plane)

    # Define the x and y coordinates of the point cloud in camera coordinates
    x = velo2cam[:, 0]
    y = velo2cam[:, 1]

    # Define the z coordinate of the point cloud in camera coordinates
    z = -(velo2cam[:, 2] + velo2cam[:, 3] * ground_plane_cam[0] + ground_plane_cam[3]) / ground_plane_cam[1]

    # Define the indices of the points that are within the image bounds
    indices = np.where((x >= 0) & (x < maxWidth) & (y >= 0) & (y < maxHeight) & (z >= min_height) & (z < max_height))[0]

    # Define the x and y coordinates of the points within the image bounds
    x = x[indices]
    y = y[indices]

    # Define the z coordinate of the points within the image bounds
    z = z[indices]

    # Define the row and column indices of the points within the image bounds
    row = np.floor((maxHeight - y) / bin_height).astype(int)
    col = np.floor(x / (maxWidth / n_bins)).astype(int)

    # Define the indices of the points that are within the bird's eye view bounds
    indices = np.where((row >= 0) & (row < n_bins) & (col >= 0) & (col < n_bins))[0]

    # Define the row and column indices of the points within the bird's eye view bounds
    row = row[indices]
    col = col[indices]

    # Define the z coordinate of the points within the bird's eye view bounds
    z = z[indices]

    # Define the bird's eye view image
    bev = np.zeros((n_bins, n_bins))
    bev[row, col] = z

    P_rect = P_rect[:, :-1]

    # Define the transform between the camera and the bird's eye view
    transform = cv2.warpPerspective(bev, P_rect, (maxWidth, maxHeight))

    return transform

def calib(calib_file, velo2cam_file):
    # Load the calibration files
    with open(calib_file, 'r') as f:
        calib_data = f.readlines()
    with open(velo2cam_file, 'r') as f:
        velo_calib_data = f.readlines()

    # Extract the calibration matrices
    P_rect = np.array(calib_data[2].strip().split(' ')[1:], dtype=np.float32).reshape(3, 4)
    R_rect = np.array(calib_data[4].strip().split(' ')[1:], dtype=np.float32).reshape(3, 3)
    T_rect = np.array([5.956621e-02, 2.900141e-04, 2.577209e-03]).reshape(3, 1)
    velo2cam = np.array(calib_data[5].strip().split(' ')[1:], dtype=np.float32).reshape(3, 4)

    return P_rect, R_rect, T_rect, velo2cam

if __name__ == "__main__":

    image = cv2.imread(image_path)

    # Load the calibration matrices
    P_rect, R_rect, T_rect, velo2cam = calib(calib_path, velo2cam_file)

    print("P_rect:")
    print(P_rect)
    print("R_rect:")
    print(R_rect)
    print("T_rect:")
    print(T_rect)
    print("velo2cam:")
    print(velo2cam)

    # Define the maximum height and width of the image
    maxHeight, maxWidth = image.shape[:2]

    # Define the ground plane
    ground_plane = [0, 1, 0, 0]

    # Calculate the transform between the camera and the xy plane of the LiDAR
    transform = project_velo2cam(ground_plane, P_rect, R_rect, T_rect, velo2cam, maxHeight, maxWidth)
    print("transform:")
    print(transform)

    # Warp the image using the transform
    warped = cv2.warpPerspective(image, transform, (maxWidth, maxHeight))

    # Display the original and warped images
    cv2.imshow("Original", image)
    cv2.imshow("Warped", warped)
    cv2.waitKey(0)




