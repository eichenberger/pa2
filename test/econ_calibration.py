import cv2
import sys
from matplotlib import pyplot as plt
import numpy as np
import argparse

class StereoCalibration(object):
    def __init__(self):
        # termination criteria
        self.criteria = (cv2.TERM_CRITERIA_EPS +
                         cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
        self.criteria_cal = (cv2.TERM_CRITERIA_EPS +
                             cv2.TERM_CRITERIA_MAX_ITER, 100, 1e-5)

        # prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
        self.objp = np.zeros((9*7, 3), np.float32)
        self.objp[:, :2] = np.mgrid[0:9, 0:7].T.reshape(-1, 2)

        # Arrays to store object points and image points from all the images.
        self.objpoints = []  # 3d point in real world space
        self.imgpoints_l = []  # 2d points in image plane.
        self.imgpoints_r = []  # 2d points in image plane.

    def read_images(self, cap1):
        SKIP= 50
        key = 0
        max = SKIP*20
        i = 0;
        while key != ord('q'):
            if i > max:
                break
            cap1.set(cv2.CAP_PROP_POS_FRAMES, i)
            i = i + SKIP
            ret, image1 = cap1.read()
            if not ret:
                break

            gray_l = cv2.extractChannel(image1, 1);
            gray_r = cv2.extractChannel(image1, 2);

            # Find the chess board corners
            ret_l, corners_l = cv2.findChessboardCorners(gray_l, (9, 7), None, cv2.CALIB_CB_ADAPTIVE_THRESH)
            if not ret_l:
                continue
            ret_r, corners_r = cv2.findChessboardCorners(gray_r, (9, 7), None, cv2.CALIB_CB_ADAPTIVE_THRESH)
            if not ret_r:
                continue
            print(f"process img {i}", file=sys.stderr)

            if ret_l and ret_r:
                # If found, add object points, image points (after refining them)
                self.objpoints.append(self.objp)

                rt = cv2.cornerSubPix(gray_l, corners_l, (11, 11),
                                      (-1, -1), self.criteria)
                self.imgpoints_l.append(corners_l)

                # Draw and display the corners
                cv2.drawChessboardCorners(gray_l, (9, 7),
                                                  corners_l, ret_l)

                rt = cv2.cornerSubPix(gray_r, corners_r, (11, 11),
                                      (-1, -1), self.criteria)
                self.imgpoints_r.append(corners_r)

                # Draw and display the corners
                cv2.drawChessboardCorners(gray_r, (9, 7),
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

        print("Starting camera calibration", file=sys.stderr)
        rt, self.M1, self.d1, self.r1, self.t1 = cv2.calibrateCamera(
            self.objpoints, self.imgpoints_l, img_shape, None, None)
        rt, self.M2, self.d2, self.r2, self.t2 = cv2.calibrateCamera(
            self.objpoints, self.imgpoints_r, img_shape, None, None)

        print("Starting stereo camrea calibration", file=sys.stderr)
        self.camera_model = self.stereo_calibrate(img_shape)

    def stereo_calibrate(self, dims):
        flags = 0
        flags |= cv2.CALIB_FIX_INTRINSIC
        flags |= cv2.CALIB_FIX_PRINCIPAL_POINT
        # flags |= cv2.CALIB_USE_INTRINSIC_GUESS
        # flags |= cv2.CALIB_FIX_FOCAL_LENGTH
        # flags |= cv2.CALIB_FIX_ASPECT_RATIO
        flags |= cv2.CALIB_ZERO_TANGENT_DIST
        # flags |= cv2.CALIB_RATIONAL_MODEL
        # flags |= cv2.CALIB_SAME_FOCAL_LENGTH
        flags |= cv2.CALIB_FIX_K3
        flags |= cv2.CALIB_FIX_K4
        flags |= cv2.CALIB_FIX_K5
        flags |= cv2.CALIB_FIX_K6

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
    parser.add_argument('camera', help='camera to use', type=str)

    np.set_printoptions(linewidth=200, suppress=True)

    args = parser.parse_args()

    cap1 = cv2.VideoCapture(args.camera)

    calib = StereoCalibration()
    calib.read_images(cap1)
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

    print("%YAML:1.0")
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

    cap1.release()

if __name__ == "__main__":
    main()
