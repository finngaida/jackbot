#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 19 21:02:18 2019

@author: claireqian
"""
import cv2 as cv
import numpy as np
import skimage
from skimage import io as skio
import matplotlib.pyplot as plt
# from utils import *
import scipy.signal
import skimage.morphology as morpho
from scipy import ndimage as ndi


def detectRegion(img):
    height, width = img.shape
    # proimg = cv.cvtColor(img,cv.COLOR_BGR2GRAY)
    # proimg = cv.GaussianBlur(proimg,(3,3),0)
    # ret, binary = cv.threshold(proimg.astype(np.uint8), 0, 255, cv.THRESH_BINARY_INV | cv.THRESH_OTSU)
    # plt.imshow(binary,cmap="gray")
    # plt.show()
    canny = cv.Canny(img, 50, 150)
    # plt.imshow(canny,cmap="gray")
    # plt.show()
    
    contours, hierarchy = cv.findContours(canny, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    
    # search in the uper-left corner
    up_rank = up_suit  = height / 2
    left_rank = left_suit = width / 2
    i_rank = 0
    i_suit = 0
    x_rank = x_suit = y_rank = y_suit = 0
    w_rank = w_suit = h_rank = h_suit = 0
    for index in range(0, len(contours)):
        x, y, w, h = cv.boundingRect(contours[index])
        if (up_rank>y and x<width/2 and y>height*0.025 and width/10>w>width/20 and height/10>h>height/20): 
            up_rank = y
            left_rank = x
            i_rank = index
            x_rank = x
            y_rank = y
            h_rank = h
            w_rank = w
    for index in range(0, len(contours)):
        if (index==i_rank): continue
        x, y, w, h = cv.boundingRect(contours[index])
        if (left_suit>x and y<height/2 and x>0.02*width and width/10>w>width/20 and height/10>h>height/20): 
            up_suit = y
            left_suit = x
            i_suit = index
            x_suit = x
            y_suit = y
            h_suit = h
            w_suit = w
    
    # codinates of value region 
    x_region = min(x_rank, x_suit)
    y_region = min(y_rank, y_suit)
    w_region = max(x_rank+w_rank, x_suit+w_suit) - x_region
    h_region = max(y_rank+h_rank, y_suit+h_suit) - y_region
    
    # draw the contour of rank and suit
    # contour_rank = cv.drawContours(img.copy(),contours,i_rank,(0,255,255),3)
    # cv.imshow("contour_rank", contour_rank)
    # contour_suit = cv.drawContours(img.copy(),contours,i_suit,(0,255,255),3)
    # cv.imshow("contour_suit", contour_suit)
    
    # draw the contour of value region in another pic
    cv.rectangle(img, (x_rank, y_rank), (x_rank+w_rank, y_rank+h_rank), (0, 0, 255), 2)
    cv.rectangle(img, (x_suit, y_suit), (x_suit + w_suit, y_suit + h_suit), (0, 255, 0), 2)
    cv.imwrite('value_regionvalue_region_3.png', img)
    # return the cordinates of the value region 
    
    return [x_rank, y_rank, w_rank, h_rank], [x_suit, y_suit, w_suit, h_suit]


def main():
    # img should be a cropped and morphed card
    img = skio.imread('test/asdf.jpg')
    a, b = detectRegion(img)
    print (a, b)


if __name__ == '__main__':
    main()
    
