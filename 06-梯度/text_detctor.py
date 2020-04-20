# -*- coding: utf-8 -*-
"""
 @File    : text_detctor.py
 @Time    : 2020/4/4 下午10:20
 @Author  : yizuotian
 @Description    :
"""
import os
import os.path as osp

import cv2


def fast_nms(boxes):
    """

    :param boxes:
    :return:
    """
    keep_boxes = []

    def _add(box):
        x, y, w, h = box
        area = w * h
        for i, cur_box in enumerate(keep_boxes):
            x1, y1, w1, h1 = cur_box
            cur_area = w1 * h1
            max_x1 = max(x, x1)
            min_x2 = min(x + w, x1 + w1)
            max_y1 = max(y, y1)
            min_y2 = min(y + h, y1 + h1)
            intersect = max(0, min_x2 - max_x1) * max(0, min_y2 - max_y1)
            if intersect > min(area, cur_area) * 0.5:
                if cur_area < area:  # 保留大的
                    keep_boxes[i] = box
                return
        keep_boxes.append(box)

    for b in boxes:
        _add(b)

    return keep_boxes


def detect(gray_img):
    """
    检测文字区域
    :param gray_img:灰度图
    :return boxes:  list of box[x,y,w,h]
    """
    gray_img = cv2.GaussianBlur(gray_img, (5, 5), 0)  # 高斯模糊
    h, w = gray_img.shape
    mser = cv2.MSER_create(_max_area=int(h * w * 0.05))
    msers, boxes = mser.detectRegions(gray_img)

    # 增加面积过滤，mser面积过滤不准确
    boxes = [[x, y, bw, bh] for x, y, bw, bh in boxes
             if 50 < bw * bh < h * w * 0.05 and
             8 <= bh < 0.8 * h and bw < 0.8 * w]
    boxes.sort(key=lambda e: e[3] * e[2], reverse=True)  # 按照面积倒序
    print("before nms len(boxes):{}".format(len(boxes)))
    boxes = fast_nms(boxes)
    print("after nms len(boxes):{}".format(len(boxes)))
    return boxes


def visual(img, boxes):
    # 画矩形框
    for box in boxes:
        x, y, w, h = box
        cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)
    return img


def run(img, debug=False):
    # img = img[5:-5, 5:-5, :]  # 去除边界
    vis = img.copy()
    # 转灰度，高斯模糊
    gray = cv2.cvtColor(vis, cv2.COLOR_BGR2GRAY)
    # mser 检测最大稳定极值区域
    boxes = detect(gray)
    vis = visual(vis, boxes)
    if debug:
        cv2.imshow('gauss', vis)
        cv2.waitKey(0)
    cv2.imwrite('../output/mser/20.jpg', vis)
    return vis


def test_dir(img_dir, out_dir):
    """
    测试目录中的图像
    :param img_dir:
    :param out_dir:
    :return:
    """
    for img_name in os.listdir(img_dir):
        img_path = osp.join(img_dir, img_name)
        img = cv2.imread(img_path)
        vis = run(img)
        out_img_path = osp.join(out_dir, img_name)
        cv2.imwrite(out_img_path, vis)


if __name__ == '__main__':
    # image = cv2.imread('../images/01.jpeg')
    # image = cv2.imread('../images/02.jpeg')
    # image = cv2.imread('../images/03.jpeg')
    # image = cv2.imread('../images/05.png')
    # image = cv2.imread('../images/20.jpg')
    # image = cv2.imread('../bill_img/02.jpeg')
    # run(image, True)
    # 测试整个目录
    # test_dir('../images', '../output/mser')
    test_dir('../img_jq', '../output/img_jq')
