# -*- coding: utf-8 -*-
"""
 @File    : table_recognize.py
 @Time    : 2020/3/23 下午9:38
 @Author  : yizuotian
 @Description    :
"""
import cv2
import numpy as np


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

    # cv2.imshow('thresh_img', thresh_img)
    # cv2.imshow('mask_img', mask_img)
    # cv2.waitKey(0)
    return mask_img


def get_table_cells(img, img_mask, border=1, min_area=1e3, max_area=1e6):
    """

    :param img:
    :param img_mask:
    :param border:
    :param min_area:
    :param max_area:
    :return:
    """
    ret, thresh = cv2.threshold(img_mask, 200, 255, cv2.THRESH_BINARY)  # 二值化
    img_erode = cv2.dilate(thresh, np.ones((3, 3)), iterations=1)

    _, contours, hierarchy = cv2.findContours(img_erode, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    roi_list = []
    coordinate_list = []  # (y1,y2,x1,x2)
    for i in range(len(contours)):
        cnt = contours[i]
        area = cv2.contourArea(cnt)
        if min_area < area < max_area:
            x, y, w, h = cv2.boundingRect(cnt)
            if w > 32 and h > 32:
                roi = img[y + border:y + h - border, x + border:x + w - border]
                roi_list.append(roi)
                coordinate_list.append([y + border, y + h - border, x + border, x + w - border])
                print(x, y, w, h)

    return roi_list, coordinate_list


if __name__ == '__main__':
    # img = cv2.imread('../images/table01.jpg')
    image = cv2.imread('../tmp/efg.jpg')
    image_mask = run(cv2.cvtColor(image, cv2.COLOR_BGR2GRAY))
    cv2.imwrite('../tmp/efg_mask.jpg', image_mask)
    cell_list, _ = get_table_cells(image, image_mask)
    for i, cell in enumerate(cell_list):
        cv2.imwrite('../tmp/out/{:03d}.jpg'.format(i), cell)
