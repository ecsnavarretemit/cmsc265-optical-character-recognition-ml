#!/usr/bin/env python

# train.py
#
# Copyright(c) Exequiel Ceasar Navarrete <esnavarrete1@up.edu.ph>
# Licensed under MIT
# Version 1.0.0-alpha1

import os
import cv2
from app import create_binary_image, imclearborder
from app.ocr import create_knowledgebase

image = os.path.join(os.getcwd(), "assets/img/training/set-1.jpg")
cv_image = cv2.imread(image)

# convert to grayscale
gray_image = cv2.cvtColor(cv_image, cv2.COLOR_BGR2GRAY)

# convert to binary image
binary_image = create_binary_image(gray_image)

# remove borders from the binary image
binary_image = imclearborder(binary_image.copy(), 50)

# pass the thresholded image and process the data
create_knowledgebase(binary_image, os.path.join(os.getcwd(), 'data/ocr'))

cv2.destroyAllWindows()


