# -*- coding: utf-8 -*-
"""
 @File    : tilt_correct.py
 @Time    : 2020/3/22 下午5:40
 @Author  : yizuotian
 @Description    :
"""
import cv2
import numpy as np


def cross_pt(line1, line2):
    """
    求两条直线交点
    :param line1: [x1,y1,x2,y2]
    :param line2: [x1,y1,x2,y2]
    :return: (x,y)
    """
    x0, y0, x1, y1 = line1
    x2, y2, x3, y3 = line2

    dx1 = x1 - x0
    dy1 = y1 - y0

    dx2 = x3 - x2
    dy2 = y3 - y2

    d1 = x1 * y0 - x0 * y1
    d2 = x3 * y2 - x2 * y3

    y = float(dy1 * d2 - d1 * dy2) / (dy1 * dx2 - dx1 * dy2)
    x = float(y * dx1 - d1) / dy1

    return x, y


def sort_pt(points):
    """

    :param points: list of (x,y)
    :return:
    """
    sp = sorted(points, key=lambda x: (int(x[1]), int(x[0])))
    # if sp[0][0] > sp[1][0]:
    #     sp[0], sp[1] = sp[1], sp[0]
    #
    # if sp[2][0] > sp[3][0]:
    #     sp[2], sp[3] = sp[3], sp[2]

    return sp


def img_corr(img):
    img_src = img.copy()
    img_gray = cv2.cvtColor(img_src, cv2.COLOR_BGR2GRAY)
    img_blur = cv2.GaussianBlur(img_gray, (3, 3), 0)
    img_canny = cv2.Canny(img_blur, 35, 189)

    # hough检测直线
    lines = cv2.HoughLinesP(img_canny, 1, np.pi / 180, threshold=30, minLineLength=200, maxLineGap=40)
    lines = lines[:, 0]  # [n,1,4]=>[n,4]

    # 画直线
    for x1, y1, x2, y2 in lines:
        cv2.line(img_src, (x1, y1), (x2, y2), (255, 255, 0), 3)

    points = np.zeros((4, 2), dtype="float32")
    points[0] = cross_pt(lines[0], lines[2])
    points[1] = cross_pt(lines[0], lines[3])
    points[2] = cross_pt(lines[1], lines[2])
    points[3] = cross_pt(lines[1], lines[3])

    sp = sort_pt(points)

    width = int(np.sqrt(((sp[0][0] - sp[1][0]) ** 2) + (sp[0][1] - sp[1][1]) ** 2))
    height = int(np.sqrt(((sp[0][0] - sp[2][0]) ** 2) + (sp[0][1] - sp[2][1]) ** 2))

    dst_rect = np.array([
        [0, 0],
        [width - 1, 0],
        [0, height - 1],
        [width - 1, height - 1]], dtype="float32")

    transform = cv2.getPerspectiveTransform(np.array(sp), dst_rect)
    warpedimg = cv2.warpPerspective(src, transform, (width, height))

    return warpedimg


if __name__ == '__main__':
    src = cv2.imread("../images/tilt04.jpg")
    dst = img_corr(src)
    cv2.imshow("Image", dst)
    cv2.waitKey(0)
    cv2.imwrite("output.jpg", dst)
