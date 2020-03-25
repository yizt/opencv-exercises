# -*- coding: utf-8 -*-
"""
 @File    : image_stitch.py
 @Time    : 2020/3/25 下午9:36
 @Author  : yizuotian
 @Description    :
"""
import cv2
import imutils


def stitch_images(image_list):
    """

    :param image_list: list of numpy
    :return:
    """
    stitcher = cv2.createStitcher() if imutils.is_cv3() else cv2.Stitcher_create()
    status, stitched = stitcher.stitch(image_list)

    if status == 0:
        cv2.imshow('stitched', stitched)
        cv2.waitKey(0)

    else:
        print('图像拼接失败')


if __name__ == '__main__':
    img_path_list = ['../images/stitch01.png', '../images/stitch02.png']
    img_list = [cv2.imread(p) for p in img_path_list]
    stitch_images(img_list)
