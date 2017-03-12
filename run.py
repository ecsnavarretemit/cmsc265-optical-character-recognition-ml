#!/usr/bin/env python

# run.py
#
# Copyright(c) Exequiel Ceasar Navarrete <esnavarrete1@up.edu.ph>
# Licensed under MIT
# Version 1.0.0-alpha2

import os
import cv2
from app import create_binary_image, imclearborder

# read the image
# image = os.path.join(os.getcwd(), "assets/img/training/set-1.jpg")
image = os.path.join(os.getcwd(), "assets/img/training/set-2.jpg")
# image = os.path.join(os.getcwd(), "assets/img/training/set-3.jpg")
# image = os.path.join(os.getcwd(), "assets/img/training/set-4.jpg")
# image = os.path.join(os.getcwd(), "assets/img/training/set-5.jpg")
# image = os.path.join(os.getcwd(), "assets/img/training/set-6.jpg")
# image = os.path.join(os.getcwd(), "assets/img/training/set-7.jpg")
cv_image = cv2.imread(image)

# convert to grayscale
gray_image = cv2.cvtColor(cv_image, cv2.COLOR_BGR2GRAY)

# convert to binary image
binary_image = create_binary_image(gray_image)

# remove borders from the binary image
binary_image = imclearborder(binary_image.copy(), 50)

# number of labels, label image, contours, GoCs
label_count, _, contours, _ = cv2.connectedComponentsWithStats(binary_image)

for label in range(1, label_count):
  x, y, w, h, size = contours[label]

  cv2.rectangle(cv_image, (x, y), (x + w, y + h), (65, 203, 62), 2)

# show the box letters
cv2.imshow('Boxed', cv2.pyrDown(cv2.pyrDown(cv_image)))

cv2.waitKey(0)
cv2.destroyAllWindows()


