import numpy as np
import cv2 as cv
import sys
from matplotlib import pyplot as plt

# Print version string
print ("OpenCV version :  {0}".format(cv.__version__))

key = 0
while key != ord('q'):
    cap = cv.VideoCapture(sys.argv[1])
    ret, img = cap.read()
    img = img[:,:,1]
    # Initiate ORB detector
    orb = cv.ORB_create()
    # find the keypoints with ORB
    kp = orb.detect(img,None)
    # compute the descriptors with ORB
    kp, des = orb.compute(img, kp)
    # draw only keypoints location,not size and orientation
    img2 = cv.drawKeypoints(img, kp, None, color=(0,255,0), flags=0)
    plt.imshow(img2), plt.show()
