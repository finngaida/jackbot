
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 19 21:02:18 2019
@author: claireqian
"""
import cv2 as cv
import numpy as np
import skimage.morphology as morpho

import utils
    
def contrast_img(img1, c, b):
    """
    To increase the contrast
    """
    rows, cols, channels = img1.shape
    blank = np.zeros([rows, cols, channels], img1.dtype)
    dst = cv.addWeighted(img1, c, blank, 1-c, b)
    return dst


def detect_region(img, show_debug=False):
    """
    Find and returns the first rank and suit regions on the given card image using Canny edge detection
    :param img: Card image should be perfectly cropped and morphed to fit the frame
    :param show_debug: Set to true to enable debug output
    :return: (x,y,w,h) rects of the suit and rank in the coordinate space of the input card respectively
    """

    height, width, ch = img.shape
    
    # increase the contrast
    a = 2
    proimg = img * float(a)
    proimg[proimg > 255] = 255
    proimg = np.round(proimg)
    proimg = proimg.astype(np.uint8)
    proimg = cv.cvtColor(img,cv.COLOR_BGR2GRAY)
    proimg = cv.GaussianBlur(proimg,(3,3),0)

    len_loc = (int) (width / 110)
    se = utils.strel('square', len_loc)
    proimg = morpho.closing(proimg, se)

    # todo: test if thresholding increases accuracy
    # ret, binary = cv.threshold(proimg, 0, 255, cv.THRESH_BINARY_INV | cv.THRESH_OTSU)

    # detect contours via Canny edge detection
    canny = cv.Canny(proimg, 50, 150)
    contours, hierarchy = cv.findContours(canny, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    
    # search in the uper-left corner
    up_rank = up_suit  = height / 4
    left_rank = left_suit = width / 4
    i_rank = 0
    i_suit = 0
    x_rank = x_suit = y_rank = y_suit = 0
    w_rank = w_suit = h_rank = h_suit = 0
    for index in range(0, len(contours)):
        x, y, w, h = cv.boundingRect(contours[index])
        if (up_rank>y and x<width/4 and width/5>w>width/20 and height/5>h>height/20):#  and cv.contourArea(contours[index])>height*width/500): 
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
        if (left_suit>x and y_rank+h_rank-2<=y<height/2 and width/10>w>width/25 and height/10>h>height/30): 
            up_suit = y
            left_suit = x
            i_suit = index
            x_suit = x
            y_suit = y
            h_suit = h
            
    
    if (x_rank<x_suit and x_suit-x_rank>0.08*width):
            w_rank = w_rank * 3
    elif (x_rank>x_suit and x_rank+w_rank-x_suit-w_suit>0.08*width):
            x_rank = int(x_rank - w_rank*3/4)
            w_rank = int(w_rank * 7 / 4)
    
    # codinates of value region 
    x_region = min(x_rank, x_suit)
    y_region = min(y_rank, y_suit)
    w_region = max(x_rank+w_rank, x_suit+w_suit) - x_region
    h_region = max(y_rank+h_rank, y_suit+h_suit) - y_region

    if show_debug:
        # draw the contour of rank and suit
        contour_rank = cv.drawContours(img.copy(),contours,i_rank,(0,255,255),3)
        # cv.imwrite("contour_rank.jpg", contour_rank)
        cv.imshow("contour_rank", contour_rank)
        contour_suit = cv.drawContours(img.copy(),contours,i_suit,(0,255,255),3)
        # cv.imwrite("contour_suit.jpg", contour_suit)
        cv.imshow("contour_suit", contour_suit)

        # draw the contour of value region in another pic

        cv.rectangle(img, (x_region, y_region), (x_region+w_region, y_region+h_region), (0, 0, 255), 2)
        cv.imshow("value region", img)
        # cv.imwrite('value_regionvalue_region_3.png', img)

        cv.rectangle(img, (x_region, y_region), (x_region+w_region, y_suit), (0, 0, 255), 2)
        cv.rectangle(img, (x_region, y_suit), (x_region + w_region, y_suit + h_suit), (0, 0, 255), 2)
        # cv.imwrite('value_region_1.png', img)
        cv.imshow("value region", img)

    # return the cordinates of the value region
    return [x_region,  y_region, w_region, h_rank], [ x_region, y_suit, w_region, h_suit]