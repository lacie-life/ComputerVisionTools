import numpy as np
import time

pointcloud_path = '/home/lacie/Datasets/KITTI/objects/train/velodyne/000366.bin'

# KITTI scanning parameters, obtained from Hough transformationRange
height = np.array(
    [0.20966667, 0.2092, 0.2078, 0.2078, 0.2078,
     0.20733333, 0.20593333, 0.20546667, 0.20593333, 0.20546667,
     0.20453333, 0.205, 0.2036, 0.20406667, 0.2036,
     0.20313333, 0.20266667, 0.20266667, 0.20173333, 0.2008,
     0.2008, 0.2008, 0.20033333, 0.1994, 0.20033333,
     0.19986667, 0.1994, 0.1994, 0.19893333, 0.19846667,
     0.19846667, 0.19846667, 0.12566667, 0.1252, 0.1252,
     0.12473333, 0.12473333, 0.1238, 0.12333333, 0.1238,
     0.12286667, 0.1224, 0.12286667, 0.12146667, 0.12146667,
     0.121, 0.12053333, 0.12053333, 0.12053333, 0.12006667,
     0.12006667, 0.1196, 0.11913333, 0.11866667, 0.1182,
     0.1182, 0.1182, 0.11773333, 0.11726667, 0.11726667,
     0.1168, 0.11633333, 0.11633333, 0.1154])
zenith = np.array([
    0.03373091, 0.02740409, 0.02276443, 0.01517224, 0.01004049,
    0.00308099, -0.00155868, -0.00788549, -0.01407172, -0.02103122,
    -0.02609267, -0.032068, -0.03853542, -0.04451074, -0.05020488,
    -0.0565317, -0.06180405, -0.06876355, -0.07361411, -0.08008152,
    -0.08577566, -0.09168069, -0.09793721, -0.10398284, -0.11052055,
    -0.11656618, -0.12219002, -0.12725147, -0.13407038, -0.14067839,
    -0.14510716, -0.15213696, -0.1575499, -0.16711043, -0.17568678,
    -0.18278688, -0.19129293, -0.20247031, -0.21146846, -0.21934183,
    -0.22763699, -0.23536977, -0.24528179, -0.25477201, -0.26510582,
    -0.27326038, -0.28232882, -0.28893683, -0.30004392, -0.30953414,
    -0.31993824, -0.32816311, -0.33723155, -0.34447224, -0.352908,
    -0.36282001, -0.37216965, -0.38292524, -0.39164219, -0.39895318,
    -0.40703745, -0.41835542, -0.42777535, -0.43621111
])
incl = -zenith


def get_range_image(pc, incl, height):
    incl_deg = incl * 180 / 3.1415
    # print(incl - np.roll(incl, 1))
    xy_norm = np.linalg.norm(pc[:, :2], ord=2, axis=1)
    error_list = []
    for i in range(len(incl)):
        h = height[i]
        theta = incl[i]
        error = np.abs(theta - np.arctan2(h - pc[:, 2], xy_norm))
        error_list.append(error)
    all_error = np.stack(error_list, axis=-1)
    row_inds = np.argmin(all_error, axis=-1)

    azi = np.arctan2(pc[:, 1], pc[:, 0])
    width = 2048
    col_inds = width - 1.0 + 0.5 - (azi + np.pi) / (2.0 * np.pi) * width
    col_inds = np.round(col_inds).astype(np.int32)
    col_inds[col_inds == width] = width - 1
    col_inds[col_inds < 0] = 0
    empty_range_image = np.full((64, width, 5), -1, dtype=np.float32)
    point_range = np.linalg.norm(pc[:, :3], axis=1, ord=2)

    order = np.argsort(-point_range)
    point_range = point_range[order]
    pc = pc[order]
    row_inds = row_inds[order]
    col_inds = col_inds[order]

    empty_range_image[row_inds, col_inds, :] = np.concatenate([point_range[:, None], pc], axis=1)

    return empty_range_image


points = np.fromfile(pointcloud_path, dtype=np.float32).reshape(-1, 4)
start = time.time()
range_image = get_range_image(points, incl, height)
end = time.time()
range_image_mask = range_image[..., 0] > -1

print("Range image generation time: ", end - start, " seconds")
print(range_image.shape)
print(range_image_mask.shape)
