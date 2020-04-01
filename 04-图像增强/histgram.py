# -*- coding: utf-8 -*-
"""
 @File    : histgram.py
 @Time    : 2020/4/1 下午5:02
 @Author  : yizuotian
 @Description    : 直方图标准化
"""
import cv2
import numpy as np


def normalize_transform(gray_img):
    '''
    :param gray_img:
    :return:
    '''
    Imin, Imax = cv2.minMaxLoc(gray_img)[:2]
    Omin, Omax = 0, 255
    # 计算a和b的值
    a = float(Omax - Omin) / (Imax - Imin)
    b = Omin - a * Imin
    out = a * gray_img + b
    out = out.astype(np.uint8)
    return out


def main(img):
    b = img[:, :, 0]
    g = img[:, :, 1]
    r = img[:, :, 2]
    b_out = normalize_transform(b)
    g_out = normalize_transform(g)
    r_out = normalize_transform(r)
    nor_out = np.stack((b_out, g_out, r_out), axis=-1)
    cv2.imshow('nor_out', np.hstack([img,nor_out]))
    cv2.waitKey()


if __name__ == '__main__':
    image = cv2.imread('../images/stitch01.png')
    main(image)
