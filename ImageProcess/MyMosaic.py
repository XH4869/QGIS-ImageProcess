# -*-coding:utf-8-*-
import math
import cv2
import numpy as np
from matplotlib import pyplot as plt


# 定义镶嵌所需参数
N = 3

Up_L_x1, Up_L_x2 = 490868, 495731
Up_L_y1, Up_L_y2 = 3410518, 3410798
SLOW_R_x1, SLOW_R_x2 = 499448, 504311
SLOW_R_y1, SLOW_R_y2 = 3395098, 3395348

width = 124
height = 506
detax = 162
detay1 = 9
detay2 = 8



def Show(M):
    plt.imshow(M)
    plt.axis("off")
    plt.show()


# 获取空白处
def GetBlank(M1, M2, M3):
    m = 0;
    n = 0;

    # Area_1
    for i in range(detay1):
        for j in range(detax):
            M3[i, j] = 8
            i += 1
            j += 1

    # Area-2、Area_3
    for i in range(detay1):
        for j in range(detax, detax + width + detax):
            M3[i, j] = M2[m, n]
            n += 1
        m += 1
        n = 0
    m = 0
    n = 0

    # Area_4、Area_7
    for i in range(detay1, detay1 + detay2 + height):
        for j in range(detax):
            M3[i, j] = M1[m, n]
            n += 1
        m += 1
        n = 0
    m = 0
    n = 0

    # Area_5
    for i in range(detay1, detay1 + height):
        for j in range(detax, detax + width):
            M3[i, j] = 255


    # Area_6
    m = detay1
    n = width
    for i in range(detay1, detay1 + height):
        for j in range(detax + width - 1, detax + width + detax):
            M3[i, j] = M2[m, n]
            n += 1
        m += 1
        n = width
    m = 0
    n = 0

    # Area_8
    m = height
    n = detax
    for i in range(height + detay1, detay1 + height + detay2):
        for j in range(detax, detax + width):
            M3[i, j] = M1[m, n]
            n += 1
        n = detax
        m += 1
    m = 0
    n = 0

    # Area_9
    for i in range(detay1 + height, detay1 + height + detay2):
        for j in range(detax + width, detax + width + detax):
            M3[i, j] = 8

    return M3



# 获取最小值 m为列表  n为double数
def GetMin(m, n):
    Min = 1000000
    k = 0

    for i in range(n):
        if m[i] < Min:
            Min = m[i]
            k = i
    return k

# 获取最佳镶嵌边   avalue为 height长度的列表
def GetBestSide(M1, M2, avalue):

    detaIK = 0

    # 以3为模板，一行能够得到122个待镶嵌点
    m = [0]
    m = m * 122

    # 模板移动寻求待镶嵌点
    for i in range(height):
        for j in range(width - 2):
            # 模板大小为3
            for k in range(-1, 2):
                detaIK = abs(int(M1[i, j + detax + 1 + k][0]) - int(M2[i + detay1, j + 1 + k][0])) + abs(int(M1[i, j + detax + 1 + k][1]) - int(M2[i + detay1, j + 1 + k][1])) + abs(int(M1[i, j + detax + 1 + k][2]) - int(M2[i + detay1, j + 1 + k][2]))
                m[j] += detaIK
        avalue[i] = GetMin(m, 122)
        for j in range(122):
            m[j] = 0


# 亮度平均和拉伸
def AverageBrightness(M1, M2, avalue):
    LAVE1 = 0
    LAVE2 = 0
    LAVE3 = 0
    RAVE1 = 0
    RAVE2 = 0
    RAVE3 = 0

    for i in range(height):
        LAVE1 += M1[i, detax + 1 + avalue[i]][0]
        LAVE2 += M1[i, detax + 1 + avalue[i]][1]
        LAVE3 += M1[i, detax + 1 + avalue[i]][2]
        RAVE1 += M2[i + detay1, 1 + avalue[i]][0]
        RAVE2 += M2[i + detay1, 1 + avalue[i]][1]
        RAVE3 += M2[i + detay1, 1 + avalue[i]][2]

    LAVE1 = LAVE1 / height
    LAVE2 = LAVE2 / height
    LAVE3 = LAVE3 / height
    RAVE1 = RAVE1 / height
    RAVE2 = RAVE2 / height
    RAVE3 = RAVE3 / height

    for i in range(detay1 + height):
        for j in range(detax + width):
            M2[i, j][0] += -int(LAVE1) + int(RAVE1)
            M2[i, j][1] += -int(LAVE2) + int(RAVE2)
            M2[i, j][2] += -int(LAVE3) + int(RAVE3)

    lMin1 = 0
    lMin2 = 0
    lMin3 = 0
    lmin1 = 500
    lmin2 = 500
    lmin3 = 500
    rMin1 = 0
    rMin2 = 0
    rMin3 = 0
    rmin1 = 500
    rmin2 = 500
    rmin3 = 500

    # fA1 = 0
    # fB1 = 0
    # fA2 = 0
    # fB2 = 0
    # fA3 = 0
    # fB3 = 0


    for i in range(height):
        if M1[i, detax + 1 + avalue[i]][0] > lMin1:
            lMin1 = M1[i, detax + 1 + avalue[i]][0]
        if M1[i, detax + 1 + avalue[i]][1] > lMin2:
            lMin2 = M1[i, detax + 1 + avalue[i]][1]
        if M1[i, detax + 1 + avalue[i]][2] > lMin3:
            lMin3 = M1[i, detax + 1 + avalue[i]][2]

        if M2[i + detay1, 1 + avalue[i]][0] > rMin1:
            rMin1 = M2[i + detay1, 1 + avalue[i]][0]
        if M2[i + detay1, 1 + avalue[i]][1] > rMin2:
            rMin2 = M2[i + detay1, 1 + avalue[i]][1]
        if M2[i + detay1, 1 + avalue[i]][2] > rMin3:
            rMin3 = M2[i + detay1, 1 + avalue[i]][2]

        if M1[i, detax + 1 + avalue[i]][0] < lmin1:
            lmin1 = M1[i, detax + 1 + avalue[i]][0]
        if M1[i, detax + 1 + avalue[i]][1] < lmin2:
            lmin2 = M1[i, detax + 1 + avalue[i]][1]
        if M1[i, detax + 1 + avalue[i]][2] < lmin3:
            lmin3 = M1[i, detax + 1 + avalue[i]][2]

        if M2[i, detax + 1 + avalue[i]][0] < rmin1:
            rmin1 = M2[i + detax, 1 + avalue[i]][0]
        if M2[i, detax + 1 + avalue[i]][1] < rmin2:
            rmin2 = M2[i + detax, 1 + avalue[i]][1]
        if M2[i, detax + 1 + avalue[i]][2] < rmin3:
            rmin3 = M2[i + detax, 1 + avalue[i]][2]

    fA1 = float(lMin1 - lmin1) / float(rMin1 - rmin1)
    fB1 = float(-fA1 * rmin1) + float(lmin1)

    fA2 = float(lMin2 - lmin2) / float(rMin2 - rmin2)
    fB2 = float(-fA2 * rmin2) + float(lmin2)

    fA3 = float(lMin3 - lmin3) / float(rMin3 - rmin3)
    fB3 = float(-fA3 * rmin3) + float(lmin3)

    # 对右图进行修改

    for i in range(detay1 + height):
        for j in range(detax + width):
            M2[i, j][0] = fA1 * M2[i, j][0] + fB1
            M2[i, j][1] = fA2 * M2[i, j][1] + fB2
            M2[i, j][2] = fA3 * M2[i, j][2] + fB3

# 裁剪 在M3中绘制重叠区部分
def Mosaic(M1, M2, M3, avalue):
    for i in range(height):
        M3[i + detay1, detax] = M1[i, detax]
        M3[i + detay1, detax + width] = M1[i + detay1, width]

        for j in range(width - 2):
            if j < avalue[i]:
                M3[i + detay1, detax + 1 + j] = M1[i, j + detax +1]
            else:
                M3[i + detay1, detax + 1 + j] = M2[i + detay1, j + 1]

    return M3


# 拼接边平滑
def Smoothen(M1,M2,M3,avalue):

    n = 5
    s = 2 * n - 1
    k = 0

    for i in range(height):
        k = avalue[i]
        for j in range(detax + width + detax):
            p1 = (k - j + (s + 1) / 2) / (s + 1)
            p2 = ((s + 1) / 2 - k + j ) / (s + 1)
            if ((j >= k - (s - 1) / 2) and (j <= k + (s - 1) / 2)):
                M3[i+detay1, j] = M1[i,j]*p1 + M2[i + detay1, j]

    return M3


# 进行影像镶嵌
def Func(M1, M2):
    w = detax + width + detax
    h = detay1 + height + detay2
    M3 = np.zeros((h, w, 3), np.uint8)

    avalue = [0]
    avalue = avalue * height


    M3 = GetBlank(M1, M2, M3)

    GetBestSide(M1, M2, avalue)

    AverageBrightness(M1, M2, avalue)

    M3 = Mosaic(M1, M2, M3, avalue)

    M3 = Smoothen(M1, M2, M3, avalue)

    result_path = "C:\\2016302590109\\Mosaic.bmp"
    cv2.imwrite(result_path, M3)
    return result_path


if __name__ == '__main__':
    print "Start."

    imageName1 ='./data/images/leftliner.bmp'
    imageName2 ='./data/images/rightliner.bmp'

    Func(imageName1, imageName2)
    print "sleep(10000)"
