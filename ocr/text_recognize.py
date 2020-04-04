# -*- coding: utf-8 -*-
"""
 @File    : text_recognize.py
 @Time    : 2020/4/4 下午10:46
 @Author  : yizuotian
 @Description    :
"""
import pytesseract
from PIL import Image


def main(img_path):
    text = pytesseract.image_to_string(Image.open(img_path))
    print(text)


if __name__ == '__main__':
    main('../images/text01.jpg')
