#!/usr/bin/env python

# test.py
#
# Copyright(c) Exequiel Ceasar Navarrete <esnavarrete1@up.edu.ph>
# Licensed under MIT
# Version 1.0.0-alpha2

import os
import cv2
import operator
import numpy as np
from app import create_binary_image, imclearborder
from app.ocr import initialize_knn_knowledge, detect_characters_by_knn, count_by_characters

image = os.path.join(os.getcwd(), "assets/img/test/set-3.jpg")
cv_image = cv2.imread(image)

# convert to grayscale
gray_image = cv2.cvtColor(cv_image, cv2.COLOR_BGR2GRAY)

# convert to binary image
binary_image = create_binary_image(gray_image)

# remove borders from the binary image
binary_image = imclearborder(binary_image.copy(), 50)

k_nearest = initialize_knn_knowledge(os.path.join(os.getcwd(), "data/ocr/matched_characters.txt"),
                                     os.path.join(os.getcwd(), "data/ocr/matched_images.txt"))

detected_character_str = detect_characters_by_knn(binary_image, cv_image, k_nearest)

# show the detected string of characters
print(detected_character_str)

stats = count_by_characters(detected_character_str)

# show the stats of the characters
for char, char_count in stats.items():
  print(f"{char}={char_count}")

# save the detected characters to a text file
with open('detected_characters.txt', 'w') as chars_f:
  chars_f.write(detected_character_str)

cv2.imshow("Boxed", cv2.pyrDown(cv2.pyrDown(cv_image)))
cv2.waitKey(0)
cv2.destroyAllWindows()


