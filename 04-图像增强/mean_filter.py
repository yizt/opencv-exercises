# -*- coding: utf-8 -*-
"""
 @File    : mean_filter.py
 @Time    : 2020/4/1 下午2:44
 @Author  : yizuotian
 @Description    :
"""

from tkinter import *

from skimage import io

im = io.imread('01.jpg', as_grey=True)
im_copy_med = io.imread('01.jpg', as_grey=True)
im_copy_mea = io.imread('01.jpg', as_grey=True)
# io.imshow(im)
for i in range(0, im.shape[0]):
    for j in range(0, im.shape[1]):
        im_copy_med[i][j] = im[i][j]
        im_copy_mea[i][j] = im[i][j]
# ui
root = Tk()
root.title("lena")
root.geometry('300x200')

medL = Label(root, text="中值滤波：")
medL.pack()
med_text = StringVar()
med = Entry(root, textvariable=med_text)
med_text.set("")
med.pack()

meaL = Label(root, text="均值滤波：")
meaL.pack()
mea_text = StringVar()
mea = Entry(root, textvariable=mea_text)
mea_text.set("")
mea.pack()


def m_filter(x, y, step):
    sum_s = []
    for k in range(-int(step / 2), int(step / 2) + 1):
        for m in range(-int(step / 2), int(step / 2) + 1):
            sum_s.append(im[x + k][y + m])
    sum_s.sort()
    return sum_s[(int(step * step / 2) + 1)]


def mean_filter(x, y, step):
    sum_s = 0
    for k in range(-int(step / 2), int(step / 2) + 1):
        for m in range(-int(step / 2), int(step / 2) + 1):
            sum_s += im[x + k][y + m] / (step * step)
    return sum_s


def on_click():
    if (med_text):
        medStep = int(med_text.get())
        for i in range(int(medStep / 2), im.shape[0] - int(medStep / 2)):
            for j in range(int(medStep / 2), im.shape[1] - int(medStep / 2)):
                im_copy_med[i][j] = m_filter(i, j, medStep)
    if (mea_text):
        meaStep = int(mea_text.get())
        for i in range(int(meaStep / 2), im.shape[0] - int(meaStep / 2)):
            for j in range(int(meaStep / 2), im.shape[1] - int(meaStep / 2)):
                im_copy_mea[i][j] = mean_filter(i, j, meaStep)
    io.imshow(im_copy_med)
    io.imsave(str(medStep) + 'med.jpg', im_copy_med)
    io.imshow(im_copy_mea)
    io.imsave(str(meaStep) + 'mea.jpg', im_copy_mea)


Button(root, text="filterGo", command=on_click).pack()

root.mainloop()
