import cv2
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
        self.objp = np.zeros((9*6, 3), np.float32)
        self.objp[:, :2] = np.mgrid[0:9, 0:6].T.reshape(-1, 2)

        # Arrays to store object points and image points from all the images.
        self.objpoints = []  # 3d point in real world space
        self.imgpoints_l = []  # 2d points in image plane.
        self.imgpoints_r = []  # 2d points in image plane.

    def read_images(self, cap1, cap2):
        key = 0
        i = 0;
        while key != ord('q'):
            cap1.set(cv2.CAP_PROP_POS_FRAMES, i)
            cap2.set(cv2.CAP_PROP_POS_FRAMES, i)
            i = i + 20
            ret, image1 = cap1.read()
            if not ret:
                break
            ret, image2 = cap2.read()
            if not ret:
                break

            img_l = image1
            img_r = image2

            gray_l = cv2.cvtColor(img_l, cv2.COLOR_BGR2GRAY)
            gray_r = cv2.cvtColor(img_r, cv2.COLOR_BGR2GRAY)

            # Find the chess board corners
            ret_l, corners_l = cv2.findChessboardCorners(gray_l, (9, 6), None, cv2.CALIB_CB_ADAPTIVE_THRESH)
            ret_r, corners_r = cv2.findChessboardCorners(gray_r, (9, 6), None, cv2.CALIB_CB_ADAPTIVE_THRESH)

            cv2.imshow("Image", img_l)
            if ret_l and ret_r:
                # If found, add object points, image points (after refining them)
                self.objpoints.append(self.objp)

                rt = cv2.cornerSubPix(gray_l, corners_l, (11, 11),
                                      (-1, -1), self.criteria)
                self.imgpoints_l.append(corners_l)

                # Draw and display the corners
                cv2.drawChessboardCorners(img_l, (9, 6),
                                                  corners_l, ret_l)

                rt = cv2.cornerSubPix(gray_r, corners_r, (11, 11),
                                      (-1, -1), self.criteria)
                self.imgpoints_r.append(corners_r)

                # Draw and display the corners
                cv2.drawChessboardCorners(img_r, (9, 6),
                                                  corners_r, ret_r)

                cv2.imshow("Image Left", img_l)
                cv2.imshow("Image Right", img_r)


            key = cv2.waitKey(1)
            if key == ord('q'):
                break
            if key == ord('a'):
                return

        img_shape = gray_r.shape
        self.shape = img_shape

        print("Starting camera calibration")
        rt, self.M1, self.d1, self.r1, self.t1 = cv2.calibrateCamera(
            self.objpoints, self.imgpoints_l, img_shape, None, None)
        rt, self.M2, self.d2, self.r2, self.t2 = cv2.calibrateCamera(
            self.objpoints, self.imgpoints_r, img_shape, None, None)

        print("Starting stereo camrea calibration")
        self.camera_model = self.stereo_calibrate(img_shape)

    def stereo_calibrate(self, dims):
        flags = 0
        flags |= cv2.CALIB_FIX_INTRINSIC
        # flags |= cv2.CALIB_FIX_PRINCIPAL_POINT
        flags |= cv2.CALIB_USE_INTRINSIC_GUESS
        flags |= cv2.CALIB_FIX_FOCAL_LENGTH
        # flags |= cv2.CALIB_FIX_ASPECT_RATIO
        flags |= cv2.CALIB_ZERO_TANGENT_DIST
        # flags |= cv2.CALIB_RATIONAL_MODEL
        # flags |= cv2.CALIB_SAME_FOCAL_LENGTH
        # flags |= cv2.CALIB_FIX_K3
        # flags |= cv2.CALIB_FIX_K4
        # flags |= cv2.CALIB_FIX_K5

        stereocalib_criteria = (cv2.TERM_CRITERIA_MAX_ITER +
                                cv2.TERM_CRITERIA_EPS, 100, 1e-5)
        ret, M1, d1, M2, d2, R, T, E, F = cv2.stereoCalibrate(
            self.objpoints, self.imgpoints_l,
            self.imgpoints_r, self.M1, self.d1, self.M2,
            self.d2, dims,
            criteria=stereocalib_criteria, flags=flags)

        print('Intrinsic_mtx_1', M1)
        print('dist_1', d1)
        print('Intrinsic_mtx_2', M2)
        print('dist_2', d2)
        print('R', R)
        print('T', T)
        print('E', E)
        print('F', F)
        print('')
        print('')

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
    parser.add_argument('camera1', help='camera to use', type=str)
    parser.add_argument('camera2', help='camera to use', type=str)

    np.set_printoptions(linewidth=200, suppress=True)

    args = parser.parse_args()

    cap1 = cv2.VideoCapture(args.camera1)
    cap2 = cv2.VideoCapture(args.camera2)

    calib = StereoCalibration()
    calib.read_images(cap1, cap2)
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
    cap2.release()

if __name__ == "__main__":
    main()
