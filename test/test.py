import cv2
from matplotlib import pyplot as plt
import numpy as np
import argparse

def main():
    parser = argparse.ArgumentParser(description='OpenCV test')
    parser.add_argument('camera', help='camera to use', type=str)

    args = parser.parse_args()

    cap = cv2.VideoCapture(args.camera)
    cap.set(cv2.CAP_PROP_EXPOSURE, 0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 752)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    key = 0
    while ord('q') != key:
        ret, image = cap.read()
        im1 = image[:,:,1]
        im2 = image[:,:,2]

        cv2.imshow("image 1", im1)
        cv2.imshow("image 2", im2)
        key = cv2.waitKey(2)

    cap.release()

if __name__ == "__main__":
    main()
