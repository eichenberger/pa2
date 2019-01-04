import cv2
import sys

fs = cv2.FileStorage(sys.argv[1], flags=cv2.FILE_STORAGE_READ)
K_l = fs.getNode("LEFT.K").mat()
K_r = fs.getNode("RIGHT.K").mat()

P_l = fs.getNode("LEFT.P").mat()
P_r = fs.getNode("RIGHT.P").mat()

R_l = fs.getNode("LEFT.R").mat()
R_r = fs.getNode("RIGHT.R").mat()

D_l = fs.getNode("LEFT.D").mat()
D_r = fs.getNode("RIGHT.D").mat()

rows_l = int(fs.getNode("LEFT.height").real())
cols_l = int(fs.getNode("LEFT.width").real())

rows_r = int(fs.getNode("RIGHT.height").real())
cols_r = int(fs.getNode("RIGHT.width").real())

cap = cv2.VideoCapture(sys.argv[2])

key = 0
while key != ord('q'):
    ret, image = cap.read()
    gray_l = cv2.extractChannel(image, 1);
    gray_r = cv2.extractChannel(image, 2);

    M1l,M2l = cv2.initUndistortRectifyMap(K_l,D_l,R_l,P_l[0:3,0:3],(cols_l,rows_l),cv2.CV_32F)
    M1r,M2r = cv2.initUndistortRectifyMap(K_r,D_r,R_r,P_r[0:3,0:3],(cols_r,rows_r),cv2.CV_32F)

    gray_l_rect = cv2.remap(gray_r,M1l,M2l,cv2.INTER_LINEAR)
    gray_r_rect = cv2.remap(gray_l,M1r,M2r,cv2.INTER_LINEAR)

    cv2.imshow("left", gray_l_rect)
    cv2.imshow("right", gray_r_rect)

    key = cv2.waitKey(0)

