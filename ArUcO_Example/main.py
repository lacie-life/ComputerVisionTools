import numpy as np
import cv2
import sys
from utils import ARUCO_DICT, rotationMatrixToQuaternion3
from calibration_store import load_coefficients
import argparse
import time

# Transform between camera
# https://stackoverflow.com/questions/58355040/how-to-transform-camera-pose-between-aruco-markers

def pose_esitmation(frame, aruco_dict_type, matrix_coefficients, distortion_coefficients):
    '''
    frame - Frame from the video stream
    matrix_coefficients - Intrinsic matrix of the calibrated camera
    distortion_coefficients - Distortion coefficients associated with your camera
    return:-
    frame - The frame with the axis drawn on it
    '''

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    cv2.aruco_dict = cv2.aruco.Dictionary_get(aruco_dict_type)
    parameters = cv2.aruco.DetectorParameters_create()

    corners, ids, rejected_img_points = cv2.aruco.detectMarkers(gray, cv2.aruco_dict, parameters)

    # If markers are detected
    if len(corners) > 0:
        for i in range(0, len(ids)):
            # Estimate pose of each marker and return the values rvec and tvec---(different from those of camera coefficients)
            rvec, tvec, markerPoints = cv2.aruco.estimatePoseSingleMarkers(corners[i], 0.02, matrix_coefficients,
                                                                           distortion_coefficients)

            # Convert rotation vector to rotation matrix
            # https://answers.ros.org/question/314828/opencv-camera-rvec-tvec-to-ros-world-pose/
            # https://docs.opencv.org/4.x/d9/d0c/group__calib3d.html#ga61585db663d9da06b68e70cfbf6a1eac
            rotationMatrix = any
            cv2.Rodrigues(rvec, rotationMatrix)

            print("Rotation Matrix")
            print(rotationMatrix)

            print("Quaternion")
            print(rotationMatrixToQuaternion3(rotationMatrix))

            print("Translation vector")
            print(tvec)

            # Draw a square around the markers
            cv2.aruco.drawDetectedMarkers(frame, corners)

            # Draw Axis
            cv2.aruco.drawAxis(frame, matrix_coefficients, distortion_coefficients, rvec, tvec, 0.01)

    return frame


if __name__ == '__main__':

    ap = argparse.ArgumentParser()
    ap.add_argument("--image", required=True, type=str, help="Path to image")
    ap.add_argument("--file", required=True, type=str, help="Path to calibration matrix")
    ap.add_argument("--type", type=str, default="DICT_ARUCO_ORIGINAL", help="Type of ArUCo tag to detect")
    args = ap.parse_args()

    if ARUCO_DICT.get(args.type, None) is None:
        print(f"ArUCo tag type '{args.type}' is not supported")
        sys.exit(0)

    aruco_dict_type = ARUCO_DICT[args.type]

    print(args.file)
    K, D = load_coefficients(args.file)

    print(K)
    print(D)

    time.sleep(2.0)

    image = cv2.imread(args.image)

    output = pose_esitmation(image, aruco_dict_type, K, D)

    cv2.imshow('Estimated Pose', output)

    cv2.waitKey(0)

    cv2.destroyAllWindows()
