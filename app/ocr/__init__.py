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




