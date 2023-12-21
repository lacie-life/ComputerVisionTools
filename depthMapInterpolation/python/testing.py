import utils_lib as pyutils
import numpy as np
import pykitti
import cv2
import time
import os
import os.path as osp
import glob
import time

def torgb(img):
    # Load RGB
    rgb_cv = np.asarray(img).copy()
    rgb_cv = cv2.cvtColor(rgb_cv, cv2.COLOR_RGB2BGR)
    rgb_cv = rgb_cv.astype(np.float32)/255
    return rgb_cv

def plot_depth(dmap_raw_np, rgb_cv, name="win"):
    rgb_cv = rgb_cv.copy()

    dmax = np.max(dmap_raw_np)
    dmin = np.min(dmap_raw_np)
    for r in range(0, dmap_raw_np.shape[0], 1):
        for c in range(0, dmap_raw_np.shape[1], 1):
            depth = dmap_raw_np[r, c]
            if depth > 0.1:
                dcol = depth/20
                rgb_cv[r, c, :] = [1-dcol, dcol, 0]
                #cv2.circle(rgb_cv, (c, r), 1, [1-dcol, dcol, 0], -1)

    cv2.namedWindow(name)
    cv2.moveWindow(name, 2500, 50)
    cv2.imshow(name, rgb_cv)
    cv2.waitKey(15)

# Parameters
basedir = "kitti"
date = "2011_09_26"
drive = "0005"
# date = "2011_10_03"
# drive = "0047"

# KITTI Load
img_dir = "../data/images/"
lidar_dir = "../data/pointclouds/"
calib_dir = "../data/calib/"
ouput_dir = "../data/dense/"

if osp.exists(ouput_dir):
    print('Output directory already exists:', ouput_dir)
else:
    os.makedirs(ouput_dir);
    print('Creating DenseIMG:', ouput_dir)

# read files
img_files = glob.glob(osp.join(img_dir, '*.png'))
lidar_files = glob.glob(osp.join(lidar_dir, '*.bin'))
calib_files = glob.glob(osp.join(calib_dir, '*.txt'))
img_files.sort()
lidar_files.sort()
calib_files.sort()


def read_calib_file(filepath):
    """Read in a calibration file and parse into a dictionary."""
    data = {}

    with open(filepath, 'r') as f:
        for line in f.readlines()[:7]:
            key, value = line.split(':', 1)
            # The only non-float values in these files are dates, which
            # we don't care about anyway
            try:
                data[key] = np.array([float(x) for x in value.split()])
            except ValueError:
                pass

    return data

for index in range(len(lidar_files)):

    img = cv2.imread(img_files[index])
    # Load Velodyne Data
    velodata = np.fromfile(lidar_files[index], dtype=np.float32).reshape(-1, 4)
    velodata[:, 3] = 1.

    calib = read_calib_file(calib_files[index])
    R_rect_00 = np.zeros((4, 4))
    R_rect_00[:3, :3] = np.array(calib['R0_rect']).reshape(-1, 3)
    R_rect_00[3, 3] = 1.

    P_rect_02 = np.zeros((4, 4))
    P_rect_02[:3, :] = np.array(calib['P2']).reshape(-1, 4)
    P_rect_02[3, 3] = 1.
    # Get intrinsics
    intr_raw = P_rect_02[:3, :3]

    vel2cam0 = np.zeros((4, 4))
    vel2cam0[:3, :] = np.array(calib['Tr_velo_to_cam']).reshape(-1, 4)
    vel2cam0[3, 3] = 1.

    start = time.time()
    # Large Image Depthmap
    large_img_size = (768/1,256/1)
    uchange = float(large_img_size[0])/float(img.shape[0])
    vchange = float(large_img_size[1])/float(img.shape[1])
    intr_large = intr_raw.copy()
    intr_large[0,:] *= uchange
    intr_large[1,:] *= vchange
    intr_large_append = np.append(intr_large, np.array([[0, 0, 0]]).T, axis=1)
    large_img = cv2.resize(torgb(img), (int(large_img_size[0]), int(large_img_size[1])), interpolation=cv2.INTER_LINEAR)
    large_params = {"filtering": 2, "upsample": 0}
    dmap_large = pyutils.generate_depth(velodata, intr_large_append, vel2cam0.astype(np.float32), int(large_img_size[0]), int(large_img_size[1]), large_params)
    end = time.time()
    print("Time:", end-start)
    plot_depth(dmap_large, large_img, "large_img")

    start = time.time()
    # Small Image Depthmap
    small_img_size = (768/4,256/4)
    uchange = float(small_img_size[0])/float(img.shape[0])
    vchange = float(small_img_size[1])/float(img.shape[1])
    intr_small = intr_raw.copy()
    intr_small[0,:] *= uchange
    intr_small[1,:] *= vchange
    intr_small_append = np.append(intr_small, np.array([[0, 0, 0]]).T, axis=1)
    small_img = cv2.resize(torgb(img), (int(small_img_size[0]), int(small_img_size[1])), interpolation=cv2.INTER_LINEAR)
    small_params = {"filtering": 0, "upsample": 0}
    dmap_small = pyutils.generate_depth(velodata, intr_small_append, vel2cam0.astype(np.float32), int(small_img_size[0]), int(small_img_size[1]), small_params)
    end = time.time()
    print("Time:", end-start)
    plot_depth(dmap_small, small_img, "small_img")

    # Upsampled
    start = time.time()
    upsampled_img_size = (768/2,256/2)
    uchange = float(upsampled_img_size[0])/float(img.shape[0])
    vchange = float(upsampled_img_size[1])/float(img.shape[1])
    intr_upsampled = intr_raw.copy()
    intr_upsampled[0,:] *= uchange
    intr_upsampled[1,:] *= vchange
    intr_upsampled_append = np.append(intr_upsampled, np.array([[0, 0, 0]]).T, axis=1)
    upsampled_img = cv2.resize(torgb(img), (int(upsampled_img_size[0]), int(upsampled_img_size[1])), interpolation=cv2.INTER_LINEAR)
    upsampled_params = {"filtering": 1, "upsample": 4}
    dmap_upsampled = pyutils.generate_depth(velodata, intr_upsampled_append, vel2cam0.astype(np.float32), int(upsampled_img_size[0]), int(upsampled_img_size[1]), upsampled_params)
    end = time.time()
    print("Time:", end-start)
    plot_depth(dmap_upsampled, upsampled_img, "upsampled_img")

    # Uniform Sampling
    start = time.time()
    uniform_img_size = (768/1,256/1)
    uchange = float(uniform_img_size[0])/float(img.shape[0])
    vchange = float(uniform_img_size[1])/float(img.shape[1])
    intr_uniform = intr_raw.copy()
    intr_uniform[0,:] *= uchange
    intr_uniform[1,:] *= vchange
    intr_uniform_append = np.append(intr_uniform, np.array([[0, 0, 0]]).T, axis=1)
    uniform_img = cv2.resize(torgb(img), (int(uniform_img_size[0]), int(uniform_img_size[1])), interpolation=cv2.INTER_LINEAR)
    uniform_params = {"filtering": 2, "upsample": 1,
                  "total_vbeams": 64, "vbeam_fov": 0.4,
                  "total_hbeams": 750, "hbeam_fov": 0.4}
    dmap_uniform = pyutils.generate_depth(velodata, intr_uniform_append, vel2cam0.astype(np.float32), int(uniform_img_size[0]), int(uniform_img_size[1]), uniform_params)
    end = time.time()
    print("Time:", end-start)
    plot_depth(dmap_uniform, uniform_img, "uniform_img")

    # Interpolation Pytorch Test

    cv2.waitKey(0)


