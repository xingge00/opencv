import cv2 as cv
import numpy as np
import scipy.io as sc


def video_demo():
    capture = cv.VideoCapture(0)
    while True:
        ret, frame = capture.read()
        frame = cv.flip(frame, 1)

        # kernel = cv.getStructuringElement(cv.MORPH_RECT, (1, 1))
        # dst = cv.erode(frame, kernel)
        # dst = cv.dilate(frame, kernel)
        cv.imshow("video", frame)
        c = cv.waitKey(60)
        if c == 27:
            break


def get_image_info(image):
    print(type(image))
    print(image.shape)
    print(image.size)
    print(image.dtype)


def access_pixels(image):
    print(image.shape)
    height = image.shape[0]
    width = image.shape[1]
    channels = image.shape[2]
    for row in range(height):
        for col in range(width):
            for c in range(channels):
                pv = image[row, col, c]
                image[row, col, c] = 255 - pv
    cv.imshow("pixels_demo", image)
    return image


def access_pixels1(image):
    print(image.shape)
    img = np.zeros([2338, 1080, 3], np.uint8)
    height = image.shape[0]
    width = image.shape[1]
    for row in range(height):
        for col in range(width):
            # img[row, col] = int(image[row][col])
            pv = image[row, col]
            img[row, col] = 255 - pv
    # cv.imshow("pixels_demo", img)
    return img


def access_pixels2(image):
    print(image.shape)
    img = np.zeros([512*3, 512*3, 3], np.uint8)
    height = image.shape[0]
    width = image.shape[1]
    tongdao = image.shape[2]
    for row in range(height):
        for col in range(width):
            for td in range(tongdao):
                img[row, col, td] = int(image[row][col][td])
    return img
    # cv.imshow("pixels_demo", img)


def access_pixels3(image):
    height = image.shape[0]
    width = image.shape[1]
    img = np.zeros([height, width], np.uint64)
    for row in range(height):
        for col in range(width):
            img[row, col] = image[row, col]
    return img


def access_pixels4(image):
    height = image.shape[0]
    width = image.shape[1]
    img = np.zeros([height, width], np.uint8)
    for row in range(height):
        for col in range(width):
            img[row, col] = image[row, col]*255
    return img


def create_image():
    img = np.zeros([25, 25], np.uint8)
    print(img.shape)
    # img[: , : , 0] = np.ones([400, 400])*127
    img.fill(1)
    # cv.imshow("pixels_demo", img)
    return img


def image_binary(image):
    print(image.shape)
    gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    rest, dst = cv.threshold(gray, 0, 255, cv.THRESH_BINARY | cv.THRESH_OTSU)
    cv.imshow("image_binary全局阈值", dst)
    # dst = cv.adaptiveThreshold(image, 255, cv.ADAPTIVE_THRESH_GAUSSIAN_C, cv.THRESH_BINARY, 85, 20)
    # cv.imshow("image_binary局部阈值", dst)
    return dst


def abc():
    cv.namedWindow("camera", 1)
    # 开启ip摄像头
    video = "http://admin:admin@192.168.1.114:8081"
    capture = cv.VideoCapture(video)

    num = 0
    while True:
        success, img = capture.read()
        cv.imshow("camera", img)

        # 按键处理，注意，焦点应当在摄像头窗口，不是在终端命令行窗口
        key = cv.waitKey(10)

        if key == 27:
            # esc键退出
            print("esc break...")
            break
        if key == ord(' '):
            # 保存一张图像
            num = num + 1
            filename = "frames_%s.jpg" % num
            cv.imwrite(filename, img)

    capture.release()
    cv.destroyWindow("camera")


def find_color(image):
    hsv = cv.cvtColor(image, cv.COLOR_BGR2HSV)
    lower_hsv = np.array([37, 43, 46])
    upper_hsv = np.array([77, 255, 255])
    mask = cv.inRange(hsv, lowerb=lower_hsv, upperb=upper_hsv)
    cv.imshow("mask", mask)


def xcorr2(M1, M2):
    # M1,M2传入两个矩阵  ps：两个矩阵要相同即M1=M2
    # M1的高
    r1 = M1.shape[0]
    # M1的宽
    c1 = M1.shape[1]
    # M2的宽
    r2 = M2.shape[0]
    # M2的高
    c2 = M2.shape[1]
    # 创建一个新的矩阵宽：r1+r2-1 高：c1+c2-1 类型看着改
    CC = np.zeros([r1+r2-1, c1+c2-1], np.uint8)
    for row in range(CC.shape[0]):
        for col in range(CC.shape[1]):
            # 创建一个空列表
            M2sum = []
            if r2 - row - 1 >= 0:
                # M2[r2 - row - 1:]意思是遍历M2的高从r2 - row - 1开始到结束
                for i in M2[r2 - row - 1:]:
                    if c2 - col - 1 >= 0:
                        for j in i[c2 - col - 1:]:
                            # print(j, end="")
                            M2sum.append(j)
                    else:
                        # 遍历i从0到c2 - col - 1
                        for j in i[:c2 - col - 1]:
                            # print(j, end="")
                            M2sum.append(j)
            else:
                for i in M2[:r2 - row - 1]:
                    if c2 - col - 1 >= 0:
                        for j in i[c2 - col - 1:]:
                            # print(j, end="")
                            M2sum.append(j)
                    else:
                        for j in i[:c2 - col - 1]:
                            # print(j, end="")
                            M2sum.append(j)
            M1sum = []
            if row - r1 + 1 <= 0:
                for i in M1[:row + 1]:
                    if col - c1 + 1 <= 0:
                        for j in i[:col + 1]:
                            # print(j, end="")
                            M1sum.append(j)
                    else:
                        for j in i[col - c1 + 1:]:
                            # print(j, end="")
                            M1sum.append(j)
            else:
                for i in M1[row - r1 + 1:]:
                    if col - c1 + 1 <= 0:
                        for j in i[:col + 1]:
                            # print(j, end="")
                            M1sum.append(j)
                    else:
                        for j in i[col - c1 + 1:]:
                            # print(j, end="")
                            M1sum.append(j)
            sum = 0
            # M1sum.__len__()获得M1sum的长度
            for i in range(M1sum.__len__()):
                sum += M1sum[i]*M2sum[i]
            # 给各个点赋值
            CC[row][col] = sum

    return CC


# 读取图片
src = cv.imread("F:/Projects/images/Lenna_rgb.tiff")
image_binary(src)
# cv.namedWindow("input image", cv.WINDOW_AUTOSIZE)
# cv.imshow("input image", src)
# find_color(src)
# dst = cv.imread("F:\Projects\images/111111111.jpg")
# # dst = cv.pyrDown(src)
# cv.imshow("dst", dst)
# gray = cv.cvtColor(src, cv.COLOR_BGR2GRAY)
# cv.imshow("gray", gray)
# edge_output = cv.Canny(gray, 50, 150)
# cv.imwrite("F:\Projects\images/test.jpg", edge_output)
# cv.imshow("edge_output", edge_output)
# print(edge_output.shape)
# fin = access_pixels1(edge_output)
# cv.imwrite("F:\Projects\images/dest.jpg", fin)

t1 = cv.getTickCount()

# 数字水印

data = sc.loadmat("F:\Projects\images\YCBCR.mat")
# q2 = data['q4_1']
# cv.imshow("q2", q2)
# q2 = cv.GaussianBlur(q1, (3, 3), 0)
# q2 = data["q2"]
# cv.imshow("q2", q2)
# print(q2)
# q3 = cv.subtract(q1, q2)
# cv.imshow("q3", q3)
# q3 = access_pixels3(q3)
# print(q3)
# # cv.imshow("q3", q3)
# q4 = xcorr2(q3, q3)
# cv.imshow("q4", q4)

# cv.imshow("q1", access_pixels1(data['q1']))
# cv.imshow("q2", access_pixels1(data['q2']))
# cv.imshow("q3", access_pixels1(data['q3']))
# cv.imshow("q4", access_pixels1(data['q4']))
# cv.imshow("q5", data['q5'])



# 遍历全部像素点
# access_pixels(src)

# 创建图片
# create_image()

# 调用摄像头
# video_demo()

# 黑白互换
# cv.bitwise_not(bin1, bin1)
t2 = cv.getTickCount()

print("time : %s ms" % ((t2-t1)/cv.getTickFrequency()*1000))


cv.waitKey(0)
cv.destroyAllWindows()
