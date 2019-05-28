import cv2 as cv
import numpy as np


def face_detect(src):
    gray = cv.cvtColor(src, cv.COLOR_BGR2GRAY)
    face_detector = cv.CascadeClassifier("F:/data/opencv/build/etc/haarcascades/haarcascade_frontalface_alt_tree.xml")
    # 参数3：越小越容易检测 错误率高
    faces = face_detector.detectMultiScale(gray, 1.1, 1)
    for x, y, w, h in faces:
        cv.rectangle(src, (x, y), (x+w, y+h), (0, 0, 255), 2)
    cv.imshow("face_detect", src)


# 读取图片
# src = cv.imread("F:\Projects\images\Lenna_rgb.tiff")
# face_detect(src)
# 读取视频
capture = cv.VideoCapture(0)
while(True):
    ret, frame = capture.read()
    frame = cv.flip(frame, 1)
    face_detect(frame)
    c = cv.waitKey(60)
    if c == 27:
        break

cv.waitKey(0)
cv.destroyAllWindows()
