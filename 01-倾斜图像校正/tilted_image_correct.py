# -*- coding: utf-8 -*-
"""
 @File    : tilted_image_correct.py
 @Time    : 2020/3/22 下午7:45
 @Author  : yizuotian
 @Description    :
"""
import cv2
import numpy as np


def rescale(x):
    x = np.abs(x)
    x = x - np.min(x)
    x = (x * 255) / np.max(x)
    return x


def gaussian_filter(img_gray):
    gaussian_lp_filter_mask = np.zeros((2700, 2700))
    d0 = 270
    m = gaussian_lp_filter_mask.shape[0]
    n = gaussian_lp_filter_mask.shape[1]
    for i in range(m):
        for j in range(n):
            power = ((i - (m / 2)) ** 2 + (j - (n / 2)) ** 2) / (2 * d0 ** 2)
            gaussian_lp_filter_mask[i, j] = np.exp(-power)

    gaussian_lp_filter_mask = gaussian_lp_filter_mask.astype(float)

    # getting image fft
    gray_image_fft = np.fft.fft2(img_gray, s=(2700, 2700), axes=(-2, -1), norm=None)

    # doing low pass filterin in frequency domain using Gaussian filter.
    filtered_image_fft = np.fft.fftshift(gray_image_fft) * gaussian_lp_filter_mask

    # Taking IFFT and obtaining low pass filtered Image.
    h, w = img_gray.shape
    filtered_image = np.fft.ifft2(filtered_image_fft, s=None, axes=(-2, -1), norm=None)
    filtered_image = rescale(filtered_image)[:h, :w].astype(np.uint8)
    return filtered_image


class LineDetection:

    def __init__(self, edges):
        self.edges = edges

    def get_hough_lines(self, min_line_len, max_line_len):
        """

        :param min_line_len:
        :param max_line_len:
        :return: lines: [N,(x1,y1,x2,y2)]
        """
        lines = cv2.HoughLinesP(self.edges, 1, np.pi / 180, 100,
                                minLineLength=min_line_len, maxLineGap=max_line_len)
        lines = lines[:, 0]  # [N,1,4] => [N,4]
        return lines

    def draw_hough_lines(self, lines):
        drawing = np.zeros(self.edges.shape, np.uint8)
        for x1, y1, x2, y2 in lines:
            cv2.line(drawing, (x1, y1), (x2, y2), (255, 255, 255), 2)

        return drawing


class CornerDetection:

    def __init__(self, lines):
        """

        :param lines:
        """
        self.lines = lines

    def get_corner_points(self):

        tl_x, tl_y = 10000, 10000
        tr_x, tr_y = -10000, 10000
        bl_x, bl_y = 10000, -10000
        br_x, br_y = -10000, -10000

        for x1, y1, x2, y2 in self.lines:
            # First Point
            if -x1 - y1 > -tl_x - tl_y:  ###  checking for min values of (x+y)
                tl_x, tl_y = x1, y1

            if -x2 - y2 > -tl_x - tl_y:
                tl_x, tl_y = x2, y2

        for x1, y1, x2, y2 in self.lines:

            if -y1 + x1 > tr_x - tr_y:
                tr_x, tr_y = x1, y1

            if -y2 + x2 > tr_x - tr_y:
                tr_x, tr_y = x2, y2

        for x1, y1, x2, y2 in self.lines:  # Third Point

            if y1 - x1 > -bl_x + bl_y:
                bl_x, bl_y = x1, y1

            if y2 - x2 > -bl_x + bl_y:
                bl_x, bl_y = x2, y2

        for x1, y1, x2, y2 in self.lines:  # Fourth Point

            if y1 + x1 > br_x + br_y:
                br_x, br_y = x1, y1

            if y2 + x2 > br_x + br_y:
                br_x, br_y = x2, y2

        # return all points in a tuple
        return tl_x, tl_y, tr_x, tr_y, bl_x, bl_y, br_x, br_y

    @classmethod
    def visualize_corner_points(cls, img, detected_points, point_size):
        tl_x, tl_y, tr_x, tr_y, bl_x, bl_y, br_x, br_y = detected_points

        cv2.circle(img, (tl_x, tl_y), point_size, (0, 0, 255), -1)
        cv2.circle(img, (tr_x, tr_y), point_size, (255, 255, 255), -1)

        cv2.circle(img, (bl_x, bl_y), point_size, (0, 255, 0), -1)
        cv2.circle(img, (br_x, br_y), point_size, (255, 0, 255), -1)

        return img


class PerspectiveTransform:
    def __init__(self, detected_points):
        self.detected_points = detected_points

    def get_rectangle(self):
        """

        :return: rectangle: (tl, tr, br, bl)
        """
        pts = np.array(self.detected_points).reshape((4, 2))

        rectangle = np.zeros((4, 2), dtype="float32")

        sum_row = pts.sum(axis=1)  # x+y
        rectangle[0] = pts[np.argmin(sum_row)]
        rectangle[2] = pts[np.argmax(sum_row)]

        diff_row = np.diff(pts, axis=1)  # y-x
        rectangle[1] = pts[np.argmin(diff_row)]
        rectangle[3] = pts[np.argmax(diff_row)]

        return rectangle

    def warp_perspective(self, image):
        rect = self.get_rectangle()
        (tl, tr, br, bl) = rect
        width_bottom = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
        width_top = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
        max_width = max(int(width_bottom), int(width_top))

        height_right = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
        height_left = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
        max_height = max(int(height_right), int(height_left))

        dst = np.array([[0, 0], [max_width - 1, 0], [max_width - 1, max_height - 1],
                        [0, max_height - 1]], dtype="float32")

        M = cv2.getPerspectiveTransform(rect, dst)
        warped = cv2.warpPerspective(image, M, (max_width, max_height))

        return warped


def main(im):
    img = im.copy()
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)  # getting gray scale from RGB image
    # img_filter = gaussian_filter(img_gray)
    img_blur = cv2.GaussianBlur(img_gray, (3, 3), 0)
    # 边缘检测
    edges = cv2.Canny(img_blur, 100, 200)
    # 直线检测
    line_detector = LineDetection(edges)
    lines = line_detector.get_hough_lines(min_line_len=300, max_line_len=366)
    hough_lines = line_detector.draw_hough_lines(lines)
    print(len(hough_lines))

    # 角检测
    corner_detector = CornerDetection(lines)  # instantiating the Corner detection
    points = corner_detector.get_corner_points()  # getting the four detected corner points
    print(points)
    plot_corner_points = corner_detector.visualize_corner_points(img, points, 26)

    # 校正
    warping = PerspectiveTransform(points)
    final_output = warping.warp_perspective(im.copy())

    cv2.imshow('gray', img_gray)
    # cv2.imshow('filter', img_filter)
    cv2.imshow('edges', edges)
    cv2.imshow('hough_lines', hough_lines)
    cv2.imshow('plot_corner_points', plot_corner_points)
    cv2.imshow('final_output', final_output)
    cv2.waitKey(0)


if __name__ == '__main__':
    image = cv2.imread('../images/tilt04.jpg')
    # h, w = image.shape[:2]
    # image = cv2.resize(image, (w // 2, h // 2))
    main(image)
