# -*- coding: utf-8 -*-
"""
 @File    : text_detctor.py
 @Time    : 2020/4/4 下午10:20
 @Author  : yizuotian
 @Description    :
"""
import cv2


def main(img):
    img = img[5:-5, 5:-5, :]  # 去除边界
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    vis = img.copy()
    # mser 检测最大稳定极值区域
    mser = cv2.MSER_create()
    msers, boxes = mser.detectRegions(gray)

    # 画凸包
    hulls = [cv2.convexHull(p.reshape(-1, 1, 2)) for p in msers]
    cv2.polylines(vis, hulls, 1, (0, 255, 0))

    # 画矩形框
    for box in boxes:
        x, y, w, h = box
        cv2.rectangle(vis, (x, y), (x + w, y + h), (255, 0, 0), 2)

    cv2.imshow('img', vis)
    cv2.waitKey(0)


if __name__ == '__main__':
    image = cv2.imread('../tmp/efg.jpg')
    main(image)
