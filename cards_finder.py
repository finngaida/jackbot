#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 20 09:03:08 2019

@author: ath5247
"""

import numpy as np
import argparse
import skimage.morphology as morpho
import matplotlib.pyplot as plt
from utils import *
import cv2


DEBUG = True


def shi(_img):
    plt.imshow(cv2.cvtColor(_img, cv2.COLOR_BGR2RGB))
    plt.show()
    

def order_points(pts):
    # initialzie a list of coordinates that will be ordered
    # such that the first entry in the list is the top-left,
    # the second entry is the top-right, the third is the
    # bottom-right, and the fourth is the bottom-left
    rect = np.zeros((4, 2), dtype = "float32")
 
    # the top-left point will have the smallest sum, whereas
    # the bottom-right point will have the largest sum
    s = pts.sum(axis = 1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]
 
    # now, compute the difference between the points, the
    # top-right point will have the smallest difference,
    # whereas the bottom-left will have the largest difference
    diff = np.diff(pts, axis = 1)
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]

    long_side = np.linalg.norm(rect[2] - rect[1])
    short_side = np.linalg.norm(rect[1] - rect[0])
    
    if short_side > long_side:
        rect = np.roll(rect, 1, axis=0)
 
    # return the ordered coordinates
    return rect
    

def four_point_transform(image, pts):
    # obtain a consistent order of the points and unpack them
    # individually
    rect = order_points(pts)
    (tl, tr, br, bl) = rect
 
    # compute the width of the new image, which will be the
    # maximum distance between bottom-right and bottom-left
    # x-coordiates or the top-right and top-left x-coordinates
    widthA = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
    widthB = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
    maxWidth = max(int(widthA), int(widthB))
 
    # compute the height of the new image, which will be the
    # maximum distance between the top-right and bottom-right
    # y-coordinates or the top-left and bottom-left y-coordinates
    heightA = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
    heightB = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
    maxHeight = max(int(heightA), int(heightB))
 
    # now that we have the dimensions of the new image, construct
    # the set of destination points to obtain a "birds eye view",
    # (i.e. top-down view) of the image, again specifying points
    # in the top-left, top-right, bottom-right, and bottom-left
    # order
    dst = np.array([
        [0, 0],
        [maxWidth - 1, 0],
        [maxWidth - 1, maxHeight - 1],
        [0, maxHeight - 1]], dtype = "float32")
 
    # compute the perspective transform matrix and then apply it
    M = cv2.getPerspectiveTransform(rect, dst)
    warped = cv2.warpPerspective(image, M, (maxWidth, maxHeight))
 
    # return the warped image
    return warped


def read_image(path):
    return cv2.imread(path)


def resize_image(img, scale=0.2):
    new_size = (int(img.shape[1] * scale), int(img.shape[0] * scale))
    return cv2.resize(img, new_size)


def select_val(img):
    img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    return img_hsv[:,:,2]


def fill_holes(img):
    img_seed = np.copy(img)
    img_seed[1:-1, 1:-1] = 255
    img_mask = img
    img_filled = morpho.reconstruction(img_seed, img_mask, method='erosion')
    
    return img_filled.astype(np.uint8)


def crop_transform(img_orig, img_bin):
    cards = []
    
    contours, _ = cv2.findContours(img_bin, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    
    contours = sorted(contours, key=cv2.contourArea, reverse=True)
    areas = [cv2.contourArea(cnt) for cnt in contours]
    cards_cnts = []
    # we assume that at least one card is present
    prev_area = areas[0]
    
    for cnt, area in zip(contours, areas):
        ratio = float(area) / prev_area
        if ratio > 0.5:
            cards_cnts.append(cnt)
            prev_area = area
        else:
            break
        
    for cnt in cards_cnts:        
        err = 0.1 * cv2.arcLength(cnt, True)
        approx = cv2.approxPolyDP(cnt, err, True)
        
        several_cards = False
        
        if (len(approx) == 4):
            approx = approx.reshape((4, 2))
            
            img_cnt = np.zeros_like(img_bin, dtype=np.uint8)
            cv2.drawContours(img_cnt, [cnt], -1, 255, cv2.FILLED)
            img_approx = np.zeros_like(img_bin, dtype=np.uint8)
            cv2.fillConvexPoly(img_approx, approx, 255)
            img_diff = np.bitwise_xor(img_approx, img_cnt)
            img_ref = np.bitwise_or(img_approx, img_cnt)
            if np.count_nonzero(img_diff) / np.count_nonzero(img_ref) > 0.1:
                several_cards = True
            else:
                found_card = four_point_transform(img_orig, approx)
                ratio = found_card.shape[0] / found_card.shape[1]
                if not np.isclose(ratio, 3.5 / 2.5, rtol=0.3):
                    several_cards = True
                else:
                    cards.append(found_card)
            
        if (len(approx) > 4 or several_cards):
            pass
            
    return cards


def select_cards(img):
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    img_filled = fill_holes(img_gray)
    img_blur = cv2.GaussianBlur(img_filled, (7, 7), 0)
    img_open = cv2.morphologyEx(img_blur, cv2.MORPH_OPEN, strel('diamond', 3))
    _, img_thr = cv2.threshold(img_open, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    
    found_cards = crop_transform(img, img_thr)      
    return found_cards
    

def main():
    # parser = argparse.ArgumentParser(description='Classify playing cards.')
    # parser.add_argument('path', metavar='path', type=str, nargs=1,
    #                     help='A path to an image')
    # args = parser.parse_args()
    
    # img = read_image(args.path)
    img = read_image("test/1.png")
    img_resized = resize_image(img)
    cards = select_cards(img_resized)
    for card in cards:
        shi(card)


if __name__ == '__main__':
    main()
