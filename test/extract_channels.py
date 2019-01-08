import cv2
import sys
from matplotlib import pyplot as plt
import numpy as np
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

def main():
    parser = argparse.ArgumentParser(description='OpenCV test')
    parser.add_argument('camera', help='camera to use', type=str)
    parser.add_argument('hidraw', help='hdiraw control device', type=str)
    parser.add_argument('out', help='output name', type=str)

    np.set_printoptions(linewidth=200, suppress=True)

    args = parser.parse_args()

    cap1 = cv2.VideoCapture(args.camera)
    cap1.set(cv2.CAP_PROP_FRAME_WIDTH, 752)
    cap1.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    set_manual_exposure(args.hidraw, 50000)

    key = 0
    i = 0

    while key != ord('q'):
        ret, image1 = cap1.read()
        gray_l = cv2.extractChannel(image1, 1);
        gray_r = cv2.extractChannel(image1, 2);

        cv2.imshow("right", gray_r)
        cv2.imshow("left", gray_l)

        key = cv2.waitKey(1)
        if key == ord('w'):
            print(f"Write image {i}")
            cv2.imwrite(args.out + str(i) + "_left.jpg", gray_l)
            cv2.imwrite(args.out + str(i) + "_right.jpg", gray_r)
            i += 1

    cap1.release()

if __name__ == "__main__":
    main()
