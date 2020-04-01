# -*- coding: utf-8 -*-
"""
 @File    : haze_remove.py
 @Time    : 2020/4/1 下午2:16
 @Author  : yizuotian 去雾
 @Description    :
"""

import argparse

import cv2
import numpy as np


def haze_removal(img, w=0.7, t0=0.1):
    # 求每个像素的暗通道
    darkChannel = img.min(axis=2)
    # 取暗通道的最大值最为全球大气光
    A = darkChannel.max()
    darkChannel = darkChannel.astype(np.double)
    # 利用公式求得透射率
    t = 1 - w * (darkChannel / A)
    # 设定透射率的最小值
    t[t < t0] = t0

    J = img
    # 对每个通道分别进行去雾
    J[:, :, 0] = (img[:, :, 0] - (1 - t) * A) / t
    J[:, :, 1] = (img[:, :, 1] - (1 - t) * A) / t
    J[:, :, 2] = (img[:, :, 2] - (1 - t) * A) / t
    return J


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('-i', '--image', required=True)
    args = vars(ap.parse_args())

    image = cv2.imread(args["image"])

    result = haze_removal(image.copy())

    cv2.imshow("HazeRemoval", np.hstack([image, result]))
    cv2.waitKey(0)


if __name__ == '__main__':
    main()
