# __init__.py
#
# Copyright(c) Exequiel Ceasar Navarrete <esnavarrete1@up.edu.ph>
# Licensed under MIT
# Version 1.0.0-alpha1

import cv2
import numpy as np

def create_binary_image(im):
  # create a copy of the image
  im = im.copy()

  # apply some bilateralFilter to remove unwanted noise
  filtered = cv2.bilateralFilter(im, 15, 75, 75)

  # then after applying some filtering, perform thresholding
  thresh = cv2.adaptiveThreshold(filtered, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2)

  # apply some morphological operation (opening and closing) to remove noises from in and out of the characters
  kernel_open = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))
  morphed = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel_open)

  kernel_close = np.ones((5, 5), np.uint8)
  morphed = cv2.morphologyEx(morphed, cv2.MORPH_CLOSE, kernel_close)

  return morphed

def imclearborder(im, radius):
  im = im.copy()

  # given a black and white image, first find all of its contours
  _, contours, _ = cv2.findContours(im, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

  # get dimensions of image
  rows, cols = im.shape

  # list of contours that touch the border
  contour_list = []

  # for each contour
  for idx in np.arange(len(contours)):
    # get the i'th contour
    cnt = contours[idx]

    # look at each point in the contour
    for pt in cnt:
      row = pt[0][1]
      col = pt[0][0]

      # if this is within the radius of the border
      # this contour will be removed
      check1 = (row >= 0 and row < radius) or (row >= rows - 1 - radius and row < rows)
      check2 = (col >= 0 and col < radius) or (col >= cols - 1 - radius and col < cols)

      if check1 or check2:
        contour_list.append(idx)
        break

  # draw black pixels on the remove contours
  for idx in contour_list:
    cv2.drawContours(im, contours, idx, (0, 0, 0), -1)

  return im


