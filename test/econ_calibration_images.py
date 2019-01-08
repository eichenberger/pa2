import cv2
import sys
from matplotlib import pyplot as plt
import numpy as np
import argparse
import os
import re

class StereoCalibration(object):
    def __init__(self, size):
        # termination criteria
        self.criteria = (cv2.TERM_CRITERIA_EPS +
                         cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
        self.criteria_cal = (cv2.TERM_CRITERIA_EPS +
                             cv2.TERM_CRITERIA_MAX_ITER, 100, 1e-5)

        # prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
        self.objp = np.zeros((8*6, 3), np.float32)
        self.objp[:, :2] = size*np.mgrid[0:8, 0:6].T.reshape(-1, 2)

        # Arrays to store object points and image points from all the images.
        self.objpoints = []  # 3d point in real world space
        self.imgpoints_l = []  # 2d points in image plane.
        self.imgpoints_r = []  # 2d points in image plane.

    def read_images(self, folder):
        i = 0;
        images = []
        images_fail = []
        for file in os.listdir(folder):
            if re.search(r'.*_left', file) == None:
                continue

            image1 = cv2.imread(folder + "/" + file)
            if image1 is None:
                break

            file_right = re.sub(r'_left', '_right', file)
            image2 = cv2.imread(folder + "/" + file_right)
            if image2 is None:
                break

            gray_l = cv2.extractChannel(image1, 1);
            gray_r = cv2.extractChannel(image2, 1);

            # Find the chess board corners
            ret_l, corners_l = cv2.findChessboardCorners(gray_l, (8, 6), None, cv2.CALIB_CB_ADAPTIVE_THRESH)
            if not ret_l:
                print("left fail")
                images_fail.append(file)
                continue
            ret_r, corners_r = cv2.findChessboardCorners(gray_r, (8, 6), None, cv2.CALIB_CB_ADAPTIVE_THRESH)
            if not ret_r:
                print("right fail")
                images_fail.append(file)
                continue

            images.append(file)
            if ret_l and ret_r:
                # If found, add object points, image points (after refining them)
                self.objpoints.append(self.objp)

                rt = cv2.cornerSubPix(gray_l, corners_l, (11, 11),
                                      (-1, -1), self.criteria)
                self.imgpoints_l.append(corners_l)

                # Draw and display the corners
                cv2.drawChessboardCorners(gray_l, (8, 6),
                                                  corners_l, ret_l)

                rt = cv2.cornerSubPix(gray_r, corners_r, (11, 11),
                                      (-1, -1), self.criteria)
                self.imgpoints_r.append(corners_r)

                # Draw and display the corners
                cv2.drawChessboardCorners(gray_r, (8, 6),
                                                  corners_r, ret_r)

                cv2.imshow("Image Left", gray_l)
                cv2.imshow("Image Right", gray_r)

            key = cv2.waitKey(1)
            if key == ord('q'):
                break
            if key == ord('a'):
                return

        img_shape = gray_r.shape
        self.shape = img_shape

        print(f"Fails: {images_fail}", file=sys.stderr)

        print("Starting camera calibration", file=sys.stderr)
        flags = 0
        # flags |= cv2.CALIB_FIX_INTRINSIC
        # flags |= cv2.CALIB_FIX_PRINCIPAL_POINT
        # flags |= cv2.CALIB_USE_INTRINSIC_GUESS
        # flags |= cv2.CALIB_FIX_FOCAL_LENGTH
        # flags |= cv2.CALIB_FIX_ASPECT_RATIO
        # flags |= cv2.CALIB_ZERO_TANGENT_DIST
        # flags |= cv2.CALIB_RATIONAL_MODEL
        # flags |= cv2.CALIB_SAME_FOCAL_LENGTH
        #flags |= cv2.CALIB_FIX_K3
        #flags |= cv2.CALIB_FIX_K4
        #flags |= cv2.CALIB_FIX_K5
        #flags |= cv2.CALIB_FIX_K6

        rt, self.M1, self.d1, self.r1, self.t1, sdi, sde, pve = cv2.calibrateCameraExtended(
            self.objpoints, self.imgpoints_l, img_shape, None, None)
        print("Reprojection error left: " + str(rt), file=sys.stderr)
        j = 0
        for image in images:
            print(f"{image}: {pve[j,0]}", file=sys.stderr)
            j+=1
        rt, self.M2, self.d2, self.r2, self.t2, sid, sde, pve = cv2.calibrateCameraExtended(
            self.objpoints, self.imgpoints_r, img_shape, None, None)
        print("Reprojection error right: " + str(rt), file=sys.stderr)
        j = 0
        for image in images:
            print(f"{image}: {pve[j,0]}", file=sys.stderr)
            j+=1

        print("Starting stereo camrea calibration", file=sys.stderr)
        self.camera_model = self.stereo_calibrate(img_shape)

    def stereo_calibrate(self, dims):
        flags = 0
        # flags |= cv2.CALIB_FIX_INTRINSIC
        # flags |= cv2.CALIB_FIX_PRINCIPAL_POINT
        # flags |= cv2.CALIB_USE_INTRINSIC_GUESS
        # flags |= cv2.CALIB_FIX_FOCAL_LENGTH
        # flags |= cv2.CALIB_FIX_ASPECT_RATIO
        #flags |= cv2.CALIB_ZERO_TANGENT_DIST
        # flags |= cv2.CALIB_RATIONAL_MODEL
        # flags |= cv2.CALIB_SAME_FOCAL_LENGTH
        #flags |= cv2.CALIB_FIX_K3
        #flags |= cv2.CALIB_FIX_K4
        #flags |= cv2.CALIB_FIX_K5
        #flags |= cv2.CALIB_FIX_K6

        stereocalib_criteria = (cv2.TERM_CRITERIA_MAX_ITER +
                                cv2.TERM_CRITERIA_EPS, 100, 1e-5)
        ret, M1, d1, M2, d2, R, T, E, F = cv2.stereoCalibrate(
            self.objpoints, self.imgpoints_l,
            self.imgpoints_r, self.M1, self.d1, self.M2,
            self.d2, dims,
            criteria=stereocalib_criteria, flags=flags)

        camera_model = dict([('M1', M1), ('M2', M2), ('D1', d1),
                            ('D2', d2), ('R1', self.r1),
                            ('R2', self.r2), ('R', R), ('T', T),
                            ('E', E), ('F', F)])

        cv2.destroyAllWindows()
        return camera_model

def print_yaml_mat(name, matrix):
    print (name + ": !!opencv-matrix")
    print (f"   rows: {matrix.shape[0]}")
    print (f"   cols: {matrix.shape[1]}")
    print ("   dt: d")
    print ("   data: " +
           np.array2string(matrix.flatten(), separator=',', suppress_small=True))


def main():
    parser = argparse.ArgumentParser(description='OpenCV test')
    parser.add_argument('images', help='images dir', type=str)
    parser.add_argument('--size', help='size of a checkerboard field', default=6, type=int)

    np.set_printoptions(linewidth=200, suppress=True)

    args = parser.parse_args()

    calib = StereoCalibration(args.size)
    calib.read_images(args.images)
    model = calib.camera_model

    R1,R2,P1,P2,Q,roi1,roi2 = cv2.stereoRectify(model['M1'],
                                    model['D1'],
                                    model['M2'],
                                    model['D2'],
                                    calib.shape,
                                    model['R'],
                                    model['T'],
                                    flags=cv2.CALIB_ZERO_DISPARITY,
                                    alpha=-1)


    T = model['T']
    print(f"Translation: {T}")

    print("%YAML:1.0")
    print(f"Camera.bf: {P2[0,3]}")
    print("")
    print(f"Camera.fx: {P1[0,0]}")
    print(f"Camera.fy: {P1[1,1]}")
    print(f"Camera.cx: {P1[0,2]}")
    print(f"Camera.cy: {P1[1,2]}")

    print("")
    print(f"Camera.width: {calib.shape[1]}")
    print(f"Camera.height: {calib.shape[0]}")

    print("")
    print(f"LEFT.height: {calib.shape[0]}")
    print(f"LEFT.width: {calib.shape[1]}")
    print_yaml_mat("LEFT.D", model['D1'])
    print_yaml_mat("LEFT.K", model['M1'])
    print_yaml_mat("LEFT.R", R1)
    print_yaml_mat("LEFT.P", P1)

    print("")
    print(f"RIGHT.height: {calib.shape[0]}")
    print(f"RIGHT.width: {calib.shape[1]}")
    print_yaml_mat("RIGHT.D", model['D2'])
    print_yaml_mat("RIGHT.K", model['M2'])
    print_yaml_mat("RIGHT.R", R2)
    print_yaml_mat("RIGHT.P", P2)

if __name__ == "__main__":
    main()
