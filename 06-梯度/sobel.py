# -*- coding: utf-8 -*-
"""
 @File    : sobel.py
 @Time    : 2020/4/20 上午10:15
 @Author  : yizuotian
 @Description    :
"""
import cv2
import numpy as np

import text_detctor


def main(img):
    gx = cv2.Sobel(img, cv2.CV_64F, dx=1, dy=0, ksize=3)
    gx = cv2.convertScaleAbs(gx)
    cv2.imshow('gx', gx)

    gy = cv2.Sobel(img, cv2.CV_64F, dx=0, dy=1, ksize=3)
    gy = cv2.convertScaleAbs(gy)
    cv2.imshow('gy', gy)

    gradient = cv2.addWeighted(gx, 0.5, gy, 0.5, 0)
    cv2.imshow('g', gradient)

    # 腐蚀和膨胀
    kernel = np.ones((3, 3), np.uint8)

    gradient = cv2.dilate(gradient, kernel, iterations=1)
    gradient = cv2.erode(gradient, kernel, iterations=1)

    cv2.imshow('dilate', gradient)

    # gradient = cv2.adaptiveThreshold(gradient,
    #                                  255,
    #                                  cv2.ADAPTIVE_THRESH_MEAN_C,  # 阈值取自相邻区域的平均值
    #                                  cv2.THRESH_BINARY,
    #                                  7,  # 领域大小
    #                                  -2)
    ret, gradient = cv2.threshold(gradient, 100, 255, cv2.THRESH_BINARY)
    cv2.imshow('thresh', gradient)

    # 检测
    boxes = text_detctor.detect(gradient)
    vis = img.copy()
    vis = text_detctor.visual(vis, boxes)

    cv2.imshow('vis', vis)

    cv2.waitKeyEx(0)


if __name__ == '__main__':
    image = cv2.imread('/Users/yizuotian/pyspace/bill_recognize/img_jq/02.jpg', 0)
    main(image)
