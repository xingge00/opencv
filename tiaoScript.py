import os
import cv2 as cv
import math
import random


def SimirarColor(p1, p2):
    r = float(p1[0])-float(p2[0])
    g = float(p1[1])-float(p2[1])
    b = float(p1[2])-float(p2[2])
    return math.sqrt(r*r+g*g+b*b)



while(True):
    os.system("adb shell screencap -p /sdcard/screen.png")
    os.system("adb pull /sdcard/screen.png")
    image = cv.imread("screen.png", cv.IMREAD_COLOR)
    # print(image.shape)
    # cv.imshow("image", image)
    Img = image[550:1350, 150:950]
    # cv.imshow("Img", Img)
    # cv.imwrite("test.png", Img)
    # 找人
    flag = 0
    pex = 0
    pey = 0
    for y in range(799, -1, -1):
        for x in range(800):
            if SimirarColor(Img[y, x], [102, 60, 54]) < 2:
                pex = x
                pey = y
                flag = 1
                break
        if flag:
            break
    print("人的坐标(%d,%d)" % (pex, pey))
    # 找点
    dx = 10
    dy = 10
    flag = 0
    if pex < 400:
        for y in range(0, pey, 1):
            for x in range(pex+80, 800, 1):
                if SimirarColor(Img[y, x], Img[100, 200]) > 9:
                    dx = x
                    dy = y
                    flag = 1
                    break
            if flag:
                break
    else:
        for y in range(0, pey, 1):
            for x in range(0, pex-80, 1):
                if SimirarColor(Img[y, x], Img[100, 100]) > 9:
                    dx = x
                    dy = y
                    flag = 1
                    break
            if flag:
                break

    print("目的坐标(%d,%d)" % (dx, dy))
    dy += 40

    d = math.sqrt((dy - pey) * (dy - pey) + (dx - pex) * (dx - pex))
    time = 1.35 * d
    a = random.randint(1, 30)
    x1 = 200 + a*3
    y1 = 300 + a*3
    x2 = 205 + a*3
    y2 = 305 + a*3
    temp = "adb shell input swipe %d %d %d %d %d" % (x1, y1, x2, y2, time)
    os.system(temp)
    t1 = random.randint(1, 8)
    print((750 + t1*250)/1000, "s")
    cv.waitKey(750 + t1*500)
