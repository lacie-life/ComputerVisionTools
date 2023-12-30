import os
import json
import numpy as np
from scipy.spatial.transform import Rotation as R
import natsort
import cv2
from pypcd4 import PointCloud
from makeBEVUtils import *
from tools import *
import torch


def project_to_image(points_3d, P):
    """Project 3D points to 2D image points."""
    num_points = points_3d.shape[1]
    points_3d_h = np.vstack([points_3d[:3, :], np.ones((1, points_3d.shape[1]))])
    points_2d_h = np.dot(P, points_3d_h)
    points_2d = points_2d_h[:2, :] / points_2d_h[2, :]
    return points_2d


def compute_2d_bbox(points_2d):
    """Compute 2D bounding box from 2D points."""
    xmin = np.min(points_2d[0, :])
    xmax = np.max(points_2d[0, :])
    ymin = np.min(points_2d[1, :])
    ymax = np.max(points_2d[1, :])
    return [xmin, ymin, xmax, ymax]


def read_label(label_filename):
    lines = [line.rstrip() for line in open(label_filename)]
    objects = [Object3d(line) for line in lines]
    return objects


def convert_to_kitti_format(json_file, output_file, img, rgb_map):
    with open(json_file, 'r') as f:
        data = json.load(f)

    P = np.array(
        data['openlabel']['streams']['s110_camera_basler_south2_8mm']['stream_properties']['intrinsics_pinhole'][
            'camera_matrix_3x4'])
    T_lidar_to_camera = np.array(
        data['openlabel']['coordinate_systems']['s110_camera_basler_south2_8mm']['pose_wrt_parent'][
            'matrix4x4']).reshape(4, 4)
    image_width = \
    data['openlabel']['streams']['s110_camera_basler_south2_8mm']['stream_properties']['intrinsics_pinhole']['width_px']
    image_height = \
    data['openlabel']['streams']['s110_camera_basler_south2_8mm']['stream_properties']['intrinsics_pinhole'][
        'height_px']

    with open(output_file, 'w') as f_out:
        for frame in data['openlabel']['frames'].values():
            for object_id, object_data in frame['objects'].items():
                val = object_data['object_data']['cuboid'].get('val', None)
                if val:
                    l = float(val[7])
                    w = float(val[8])
                    h = float(val[9])

                    quat_x = float(val[3])
                    quat_y = float(val[4])
                    quat_z = float(val[5])
                    quat_w = float(val[6])
                    roll, pitch, yaw = R.from_quat([quat_x, quat_y, quat_z, quat_w]).as_euler("xyz", degrees=True)
                    yaw = np.arctan2(np.sin(yaw), np.cos(yaw))

                    location = np.array(
                        [
                            [float(val[0])],
                            [float(val[1])],
                            [float(val[2])],
                        ]
                    )

                    points_3d = np.array([[location[0][0] - l / 2, location[0][0] - l / 2, location[0][0] + l / 2,
                                           location[0][0] + l / 2, location[0][0] - l / 2, location[0][0] - l / 2,
                                           location[0][0] + l / 2, location[0][0] + l / 2],
                                          [location[1][0] - w / 2, location[1][0] + w / 2, location[1][0] + w / 2,
                                           location[1][0] - w / 2, location[1][0] - w / 2, location[1][0] + w / 2,
                                           location[1][0] + w / 2, location[1][0] - w / 2],
                                          [location[2][0] - h / 2, location[2][0] - h / 2, location[2][0] - h / 2,
                                           location[2][0] - h / 2, location[2][0] + h / 2, location[2][0] + h / 2,
                                           location[2][0] + h / 2, location[2][0] + h / 2]])

                    print(points_3d.shape)
                    print(points_3d)
                    points_3d = np.vstack([points_3d, np.ones((1, points_3d.shape[1]))])
                    points_3d = np.dot(T_lidar_to_camera, points_3d)

                    location = np.vstack([location, np.ones((1, location.shape[1]))])
                    location = np.dot(T_lidar_to_camera, location)

                    print(location)

                    points_2d = project_to_image(points_3d, P)

                    if np.any(points_2d[0, :] < 0) or np.any(points_2d[0, :] > image_width) or np.any(
                            points_2d[1, :] < 0) or np.any(points_2d[1, :] > image_height):
                        continue  # Skip this object if it's not in the camera's FOV

                    bbox_2d = compute_2d_bbox(points_2d)

                    alpha = -np.arctan2(-points_2d[0, :].mean(), points_2d[1, :].mean()) + yaw

                    f_out.write(
                        f"{object_data['object_data']['type']} 0.0 0 {alpha} {bbox_2d[0]} {bbox_2d[1]} {bbox_2d[2]} {bbox_2d[3]} {h} {w} {l} {location[0][0]} {location[1][0]} {location[2][0]} {yaw} 0\n")

                    cv2.rectangle(img, (int(bbox_2d[0]), int(bbox_2d[1])), (int(bbox_2d[2]), int(bbox_2d[3])),
                                  color=(255, 0, 0), thickness=2)
                    # cv2.imshow("gs",img)
                    # cv2.imshow("rgb_map", rgb_map)
                    # cv2.waitKey(0)
    return img


# json_origin_path = '/mnt/hdd/A9_dataset/R02/a9_dataset_r02_s04/labels_point_clouds/s110_lidar_ouster_south'
# json_file_names = natsort.natsorted(os.listdir(json_origin_path))
# save_txt_path = '/home/jiwon/3d_object_detection/Complex-YOLOv4-Pytorch/dataset/A9_dataset/training/label_2'


# for json_file_name in json_file_names:
#     json_file_path = os.path.join(json_origin_path,json_file_name)
#     output_file = os.path.join(save_txt_path,json_file_name[:-5]+".txt")
#     convert_to_kitti_format(json_file_path, output_file)

json_file = '/mnt/hdd/A9_dataset/R02/a9_dataset_r02_s01/labels_point_clouds/s110_lidar_ouster_south/1646667310_150192886_s110_lidar_ouster_south.json'
output_file = 'kitti_format_data.txt'
img_path = "/mnt/hdd/A9_dataset/R02/a9_dataset_r02_s01/images/s110_camera_basler_south2_8mm/1646667310_139995078_s110_camera_basler_south2_8mm.jpg"
pc_path = "/mnt/hdd/A9_dataset/R02/a9_dataset_r02_s01/point_clouds/s110_lidar_ouster_south/1646667310_150192886_s110_lidar_ouster_south.pcd"

pc = PointCloud.from_path(pc_path).numpy()
pc = pc[:, :4]

pc_reduce = removePoints(pc, boundary)

rgb_map = makeBVFeature(pc_reduce, DISCRETIZATION, boundary)

rgb_map = np.transpose(rgb_map, (1, 2, 0))

img = cv2.imread(img_path)

img = convert_to_kitti_format(json_file, output_file, img, rgb_map)

print(rgb_map.shape)

objs = read_label("/home/jiwon/3d_object_detection/tum-traffic-dataset-dev-kit/src/visualization/kitti_format_data.txt")

labels, _ = read_labels_for_bevbox(objs)


labels[:, 1:] = camera_to_lidar_box(labels[:, 1:])  # convert rect cam to velo cord


target = build_yolo_target(labels)

n_target = len(target)
targets = torch.zeros((n_target, 8))

targets[:, 1:] = torch.from_numpy(target)

targets[:, 2:6] *= 608
# Get yaw angle
targets[:, 6] = torch.atan2(targets[:, 6], targets[:, 7])

print(rgb_map.shape)

for c, x, y, w, l, yaw in targets[:, 1:7].numpy():
    # Draw rotated box
    print("===============================")
    print(c, x, y, w, l, yaw)
    rgb_map = drawRotatedBox(rgb_map, x, y, w, l, yaw, colors[int(c)])

rgb_map = cv2.rotate(rgb_map, cv2.ROTATE_180)

cv2.imshow("rgb_map", rgb_map.astype(np.float32))

cv2.imshow("img", img)

cv2.waitKey(0)

