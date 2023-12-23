import os
import json
import numpy as np
from scipy.spatial.transform import Rotation as R

def project_to_image(points_3d, P):
    """Project 3D points to 2D image points."""
    num_points = points_3d.shape[1]
    points_3d_h = np.vstack([points_3d, np.ones((1, num_points))])
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

def convert_to_kitti_format(json_file, output_file):
    with open(json_file, 'r') as f:
        data = json.load(f)

    P = np.array(data['openlabel']['streams']['s110_camera_basler_south1_8mm']['stream_properties']['intrinsics_pinhole']['camera_matrix_3x4'])

    with open(output_file, 'w') as f_out:
        for frame in data['openlabel']['frames'].values():
            for object_id, object_data in frame['objects'].items():
                val = object_data['object_data']['cuboid'].get('val', None)
                if val:
                    h = float(val[7])
                    w = float(val[8])
                    l = float(val[9])

                    quat_x = float(val[3])
                    quat_y = float(val[4])
                    quat_z = float(val[5])
                    quat_w = float(val[6])
                    roll, pitch, yaw = R.from_quat([quat_x, quat_y, quat_z, quat_w]).as_euler("xyz", degrees=True)

                    location = np.array(
                        [
                            [float(val[0])],
                            [float(val[1])],
                            [float(val[2])],
                        ]
                    )

                    points_3d = np.array([[location[0][0]-l/2, location[0][0]-l/2, location[0][0]+l/2, location[0][0]+l/2, location[0][0]-l/2, location[0][0]-l/2, location[0][0]+l/2, location[0][0]+l/2],
                                          [location[1][0]-w/2, location[1][0]+w/2, location[1][0]+w/2, location[1][0]-w/2, location[1][0]-w/2, location[1][0]+w/2, location[1][0]+w/2, location[1][0]-w/2],
                                          [location[2][0]-h/2, location[2][0]-h/2, location[2][0]-h/2, location[2][0]-h/2, location[2][0]+h/2, location[2][0]+h/2, location[2][0]+h/2, location[2][0]+h/2]])

                    points_2d = project_to_image(points_3d, P)
                    bbox_2d = compute_2d_bbox(points_2d)

                    f_out.write(f"{object_data['object_data']['type']} 0.0 0 -10.0 {bbox_2d[0]} {bbox_2d[1]} {bbox_2d[2]} {bbox_2d[3]} {h} {w} {l} {location[0][0]} {location[1][0]} {location[2][0]} {yaw} 0\n")

json_file = '1646667310_053239541_s110_lidar_ouster_south.json'
output_file = 'kitti_format_data.txt'
convert_to_kitti_format(json_file, output_file)



