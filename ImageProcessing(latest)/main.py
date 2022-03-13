# -*- coding: utf-8 -*-
# @Author : lifz
# @Time : 2022/3/6
# @File : main.py
# @Mail : leoleechn@outlook.com

import cv2 as cv
import numpy as np
import csv
import matplotlib.pyplot as plt


def main():
    # 读取图像
    img1 = cv.imread("Img_source.jpg")
    img2 = cv.imread("Img_source.jpg")
    img3 = cv.imread("Img_source.jpg")

    Img_h = img1.shape[0]
    Img_w = img1.shape[1]
    print('-' * 20)
    print("图片大小:" + str(Img_h) + "px*" + str(Img_w) + "px")
    print('-' * 20 + '\n')

    imgProcessing(img1, img2, img3)

    imgTemp1 = cv.imread("Img_source.jpg")
    imgTemp2 = cv.imread("Img_thres.jpg")
    imgTemp3 = cv.imread("Img_result.jpg")
    imgTemp4 = cv.imread("Img_plane.jpg")

    plt.subplot(221)
    # plt.title("Img_source")
    plt.imshow(imgTemp1)
    # plt.xticks([])
    # plt.yticks([])
    plt.subplot(222)
    # plt.title("Img_thres")
    plt.imshow(imgTemp2)
    # plt.xticks([])
    # plt.yticks([])
    plt.subplot(223)
    # plt.title("Img_result")
    plt.imshow(imgTemp3)
    # plt.xticks([])
    # plt.yticks([])
    plt.subplot(224)
    # plt.title("Img_plane")
    plt.imshow(imgTemp4)
    # plt.xticks([])
    # plt.yticks([])
    plt.savefig("result_1.jpg")
    plt.show()

    # plt.imshow(imgTemp1)
    # plt.grid()
    # plt.show()

    gridProcessing(img1)

    cv.waitKey(0)  # 延时
    cv.destroyAllWindows()  # 释放内存


# 图像处理
def imgProcessing(var1, var2, var3):
    # 中值滤波
    Img_temp = cv.medianBlur(var1, 15)

    # 图像差分
    Img_diff = cv.absdiff(Img_temp, var2)
    # cv.imshow("Img_diff", Img_diff)  # 差分图

    # 二值化
    # Img_gray = cv.cvtColor(Img_diff, cv.COLOR_BGR2GRAY)
    # _, Img_thres = cv.threshold(Img_gray, 24, 255, cv.THRESH_BINARY)
    # cv.imshow("Img_thres", Img_thres)  # 二值化处理图

    # cv.imwrite("Img_thres.jpg", Img_thres)

    # 查找轮廓
    contours, hierarchy = cv.findContours(noiseProcessing(Img_diff), cv.RETR_EXTERNAL, cv.CHAIN_APPROX_NONE)

    # 输出缺陷个数
    print('-' * 20)
    print("不可忽略缺陷个数:" + str(len(contours)))
    print('-' * 20)

    for i in range(0, len(contours)):
        length = cv.arcLength(contours[i], True)
        # 通过缺陷轮廓长度筛选
        if length > 16:
            cv.drawContours(var2, contours[i], -1, (0, 0, 0), 2)

    # cv.imshow("Img_result", var2)  # 结果图

    cv.imwrite("Img_result.jpg", var2)

    planeDisplay(contours, var3)

    # Img_gray = cv.cvtColor(Img_diff, cv.COLOR_BGR2GRAY)
    # norm_img = np.zeros(Img_gray.shape)
    # cv.normalize(Img_gray, norm_img, 0, 255, cv.NORM_MINMAX)
    # norm_img = np.asarray(norm_img, dtype=np.uint8)
    # heat_img = cv.applyColorMap(norm_img, cv.COLORMAP_JET)  # 注意此处的三通道热力图是cv2专有的GBR排列
    # heat_img = cv.cvtColor(heat_img, cv.COLOR_BGR2RGB)  # 将BGR图像转为RGB图像
    # img_add = cv.addWeighted(cv.imread("Img_source.jpg"), 0.3, heat_img, 0.7, 0)
    # cv.imshow("Img_heat", img_add)


def noiseProcessing(var1):
    # 图像进行高斯模糊降噪，去除噪声干扰，然后再二值化
    blurred = cv.GaussianBlur(var1, (15, 15), 0)
    gray = cv.cvtColor(blurred, cv.COLOR_BGR2GRAY)
    _, binary = cv.threshold(gray, 0, 255, cv.THRESH_BINARY | cv.THRESH_OTSU)
    # cv.imshow("Img_thres", binary)  # 二值化处理图
    cv.imwrite("Img_thres.jpg", binary)

    # 图像均值迁移去噪声，然后二值化处理
    # blurred = cv.pyrMeanShiftFiltering(binary, 18, 20)
    # gray = cv.cvtColor(blurred, cv.COLOR_BGR2GRAY)
    # _, binary = cv.threshold(gray, 0, 255, cv.THRESH_BINARY | cv.THRESH_OTSU)
    # cv.imshow("Img_thres", binary)  # 二值化处理图

    return binary


# 平面几何关系处理
def planeDisplay(var1, var2):
    # 寻找边界坐标
    for c in var1:
        # 计算点集最外面的矩形边界
        x, y, w, h = cv.boundingRect(c)
        # 绘制边界矩形
        cv.rectangle(var2, (x, y), (x + w, y + h), (0, 255, 0), 1)
        # 寻找面积最小的矩形
        rect = cv.minAreaRect(c)
        # 得到最小矩形的坐标
        box = cv.boxPoints(rect)
        # 标准化坐标到整数
        box = np.int0(box)
        # 画出边界
        cv.drawContours(var2, [box], 0, (0, 0, 255), 1)
        # 坐标输出
        # print(box)

        # 坐标写入csv文件
        file = open("data.csv", mode='a', encoding="utf-8")
        writer = csv.writer(file)
        writer.writerow(box)
        file.close()

    # cv.imshow("Img_plane", var2)  # 平面处理图

    cv.imwrite("Img_plane.jpg", var2)


# 网格化处理
def gridProcessing(var1):
    # imgTemp = cv.imread("Img_source.jpg")
    # imgWidth = imgTemp.size[0]
    # imgHeight = imgTemp.size[1]
    # imgTemp = imgTemp.convert("RGB")
    # array = []
    # for x in range(imgWidth):
    #     for y in range(imgHeight):
    #         r, g, b = imgTemp.getpixel((x, y))
    #         rgb = (hex(r), hex(g), hex(b))
    #         array.append(rgb)
    # print(array)

    bit_1 = 40
    bit_2 = 27

    imgTemp = var1
    imgWidth = int(imgTemp.shape[1] / bit_1)
    imgHeight = int(imgTemp.shape[0] / bit_2)

    # for j in range(36):
    #     for i in range(36):
    #         box = (imgWidth * i, imgHeight * j, imgWidth * (i + 1), imgHeight * (j + 1))
    #         region = imgTemp.crop(box)
    #         region.save("oh.jpg".format(j, i))

    # 创建新的图像
    imgNew = np.zeros((imgTemp.shape[0], imgTemp.shape[1], 3), np.uint8)
    # 图像循环采样
    for i in range(bit_2):
        # 获取y坐标
        y = i * imgHeight
        for j in range(bit_1):
            # 获取x坐标
            x = j * imgWidth
            # 获取填充颜色,左上角像素点
            b = imgTemp[y, x][0]
            g = imgTemp[y, x][1]
            r = imgTemp[y, x][2]

            # 循环设置小区域采样
            for n in range(imgHeight):
                for m in range(imgWidth):
                    imgNew[y + n, x + m][0] = np.uint8(b)
                    imgNew[y + n, x + m][1] = np.uint8(g)
                    imgNew[y + n, x + m][2] = np.uint8(r)

    # cv.imshow("sampling", imgNew)  # 采样量化图
    cv.imwrite("sampling.jpg", imgNew)

    # 绘制网格
    for i in range(12, 480, imgWidth):
        cv.line(imgNew, (i, 0), (i, 324), (0, 0, 0), 1)
    for j in range(12, 324, imgHeight):
        cv.line(imgNew, (0, j), (480, j), (0, 0, 0), 1)
    # cv.imshow("gridSampling", imgNew)
    cv.imwrite("gridSampling.jpg", imgNew)

    imgTemp1 = cv.imread("sampling.jpg")
    imgTemp2 = cv.imread("gridSampling.jpg")
    plt.subplot(121)
    plt.imshow(imgTemp1)
    plt.subplot(122)
    plt.imshow(imgTemp2)
    plt.savefig("result_2.jpg")
    plt.show()

    # 热力图绘制
    imgTemp3 = cv.medianBlur(var1, 15)
    imgDiff = cv.absdiff(imgTemp3, imgNew)
    imgGray = cv.cvtColor(imgDiff, cv.COLOR_BGR2GRAY)
    # imgHeat = cv.applyColorMap(cv.imread("Img_thres.jpg"), cv.COLORMAP_JET)
    imgHeat = cv.applyColorMap(imgGray, cv.COLORMAP_JET)
    cv.imwrite("heat.jpg", imgHeat)
    plt.imshow(cv.imread("heat.jpg"))
    plt.colorbar()
    plt.savefig("result_3.jpg")
    plt.show()


if __name__ == "__main__":
    main()
