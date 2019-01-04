#!/usr/bin/env python

'''
Simple example of stereo image matching and point cloud generation.

Resulting .ply file cam be easily viewed using MeshLab ( http://meshlab.sourceforge.net/ )
'''

import numpy as np
import cv2
import argparse
import io

def set_manual_exposure(hidraw, value):
    MAX_EXPOSURE=300000
    if value >= MAX_EXPOSURE:
        print(f'Exposure must be less than {MAX_EXPOSURE} (is {value})')
        return
    f = io.open(hidraw, 'wb', buffering=0)
    data = bytes([0x78, 0x02, (value >> 24)&0xFF, (value >> 16)&0xFF, (value>>8)&0xFF, value&0xFF])
    f.write(data)
    f.close()

def set_auto_exposure(hidraw):
    set_manual_exposure(hidraw, 1)

def read_calib(file):
    fs = cv2.FileStorage(file, flags=cv2.FILE_STORAGE_READ)
    cal_l = {}
    cal_r = {}
    cal_l['K'] = fs.getNode("LEFT.K").mat()
    cal_r['K'] = fs.getNode("RIGHT.K").mat()

    cal_l['P'] = fs.getNode("LEFT.P").mat()
    cal_r['P'] = fs.getNode("RIGHT.P").mat()

    cal_l['R'] = fs.getNode("LEFT.R").mat()
    cal_r['R'] = fs.getNode("RIGHT.R").mat()

    cal_l['D'] = fs.getNode("LEFT.D").mat()
    cal_r['D'] = fs.getNode("RIGHT.D").mat()

    cal_l['rows'] = int(fs.getNode("LEFT.height").real())
    cal_l['cols'] = int(fs.getNode("LEFT.width").real())

    cal_r['rows'] = int(fs.getNode("RIGHT.height").real())
    cal_r['cols'] = int(fs.getNode("RIGHT.width").real())
    return cal_l, cal_r

def rectify(cal):
    M1,M2 = cv2.initUndistortRectifyMap(cal['K'],cal['D'],cal['R'],cal['P'][0:3,0:3],(cal['cols'],cal['rows']),cv2.CV_32F)

    return M1, M2

def mouseHandler(event, x, y, flags, data):
    if event == cv2.EVENT_LBUTTONDOWN:
        print(f'Distance at {x}x{y}: {data.depth[y,x]}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='OpenCV test')
    parser.add_argument('camera', help='camera to use', type=str)
    parser.add_argument('calib', help='calibration file', type=str)
    parser.add_argument('hidraw', help='hdiraw control device', type=str)

    args = parser.parse_args()

    cal_l, cal_r = read_calib(args.calib)
    M1l, M2l = rectify(cal_l)
    M1r, M2r = rectify(cal_r)

    cap1 = cv2.VideoCapture(args.camera)

    set_manual_exposure(args.hidraw, 3000)

    key = 0
    bf = 45.1932

    class DepthHolder:
        def __init__(self):
            self.depth = None


    depth_holder = DepthHolder()
    cv2.namedWindow("disparity");
    cv2.namedWindow("left");
    cv2.setMouseCallback("disparity", mouseHandler, depth_holder)
    cv2.setMouseCallback("left", mouseHandler, depth_holder)

    # disparity range is tuned for 'aloe' image pair
    window_size = 3
    min_disp = 16
    num_disp = 112-min_disp
    stereo = cv2.StereoSGBM_create(minDisparity = min_disp,
        numDisparities = num_disp,
        blockSize = 16,
        P1 = 8*3*window_size**2,
        P2 = 32*3*window_size**2,
        disp12MaxDiff = 1,
        uniquenessRatio = 10,
        speckleWindowSize = 100,
        speckleRange = 32
    )

    stereo_right = cv2.ximgproc.createRightMatcher(stereo)
    wlsfilter = cv2.ximgproc.createDisparityWLSFilter(stereo)
    wlsfilter.setLambda(800.0)
    wlsfilter.setSigmaColor(0.8)

    while key != ord('q'):
        ret, image = cap1.read()
        gray_l = cv2.extractChannel(image, 1);
        gray_r = cv2.extractChannel(image, 2);

        imgL = cv2.remap(gray_l, M1l, M2l, cv2.INTER_LINEAR)
        imgR = cv2.remap(gray_r, M1r, M2r, cv2.INTER_LINEAR)


        displ = stereo.compute(imgL, imgR).astype(np.float32)
        dispr = stereo_right.compute(imgR, imgL).astype(np.float32)

        disp = displ.copy()
        disp = wlsfilter.filter(displ, imgL, disp, disparity_map_right = dispr) / 16.0
        depth = bf/disp
        depth_holder.depth = depth

        cv2.imshow('left', gray_l)
        cv2.imshow('right', gray_r)
        cv2.imshow('disparity', (disp-min_disp)/num_disp)
        key = cv2.waitKey(1)

    cap1.release()
    cv2.destroyAllWindows()
