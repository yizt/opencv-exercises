# -*- coding: utf-8 -*-
"""
 @File    : table_recognize.py
 @Time    : 2020/3/23 下午9:38
 @Author  : yizuotian
 @Description    :
"""
import cv2


def run(gray_img):
    thresh_img = cv2.adaptiveThreshold(~gray_img,
                                       255,
                                       cv2.ADAPTIVE_THRESH_MEAN_C,  # 阈值取自相邻区域的平均值
                                       cv2.THRESH_BINARY,
                                       15,  # 领域大小
                                       -2)
    h_img = thresh_img.copy()
    v_img = thresh_img.copy()

    scale = 15
    h_size = int(h_img.shape[1] / scale)

    h_structure = cv2.getStructuringElement(cv2.MORPH_RECT, (h_size, 1))  # 形态学因子
    h_erode_img = cv2.erode(h_img, h_structure, 1)

    h_dilate_img = cv2.dilate(h_erode_img, h_structure, 1)
    v_size = int(v_img.shape[0] / scale)

    v_structure = cv2.getStructuringElement(cv2.MORPH_RECT, (1, v_size))  # 形态学因子
    v_erode_img = cv2.erode(v_img, v_structure, 1)
    v_dilate_img = cv2.dilate(v_erode_img, v_structure, 1)

    mask_img = h_dilate_img + v_dilate_img

    cv2.imshow('thresh_img', thresh_img)
    cv2.imshow('mask_img', mask_img)

    cv2.waitKey(0)


if __name__ == '__main__':
    img = cv2.imread('../images/table01.jpg')
    run(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY))
