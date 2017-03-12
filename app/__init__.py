# __init__.py
#
# Copyright(c) Exequiel Ceasar Navarrete <esnavarrete1@up.edu.ph>
# Licensed under MIT
# Version 1.0.0-alpha1

import os
import sys
import cv2
import glob
import numpy as np

def create_cv_im_instance(image_path):
  return {
    'path': image_path,
    'cv_im': cv2.imread(image_path)
  }

# TODO: replace sys.exists(1) with a custom exception
def create_cv_im_instances_from_dir(image_dir_path, **kwargs):
  file_exts = kwargs.get('file_exts', ['jpg', 'png'])

  if not os.path.exists(image_dir_path):
    print(f'Directory of Images: {image_dir_path} does not exist')
    sys.exit(1)

  # get all images in the directory that matches the extensions provided
  images = []
  for ext in file_exts:
    images.extend(glob.glob(f"{image_dir_path}/*.{ext}"))

  # terminate if no images are found
  if len(images) == 0:
    print(f'No images in the source directory {image_dir_path}')
    sys.exit(1)

  # convert images list to cv image instances list
  return list(map(create_cv_im_instance, images))

# TODO: "can be possibly" made more cleaner by using function currying and map-reduce
def matches_contours(contours_collection, contour, **kwargs):
  threshold = kwargs.get('threshold', 0.4)
  comparison_method = kwargs.get('comparison_method', 1)
  comparison_method_param = kwargs.get('comparison_method_param', 0)

  # return flag
  flag = True

  for prev_contour in contours_collection:
    probability = cv2.matchShapes(contour, prev_contour, comparison_method, comparison_method_param)

    # break out immediately since one of them is above the threshold
    if probability > threshold:
      flag = False
      break

  return flag

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


