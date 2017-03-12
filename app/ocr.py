# ocr.py
#
# Copyright(c) Exequiel Ceasar Navarrete <esnavarrete1@up.edu.ph>
# Licensed under MIT
# Version 1.0.0-alpha1

import os
import sys
import cv2
import string
import numpy as np

# In ASCII numbers (0-9 and lowercase/uppercase letters)
VALID_CHARACTERS = [ord(char) for char in string.digits + string.ascii_letters]

# Resized image dimensions
RESIZED_IMAGE_WIDTH = 30
RESIZED_IMAGE_HEIGHT = 30

# TODO: add checking if the passed image is a grayscale image
# TODO: replace sys.exists(1) with a custom exception
def create_knowledgebase(im, data_dst):
  if not os.path.exists(data_dst):
    print(f'Path: {data_dst} does not exist')
    sys.exit(1)

  # copy the image instance to prevent any modifications
  cv_im = im.copy()

  # we only need the contours, just discard the first and last elements returned by the function
  _, contours, _ = cv2.findContours(cv_im, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

  matched_characters = []
  matched_images = np.empty((0, RESIZED_IMAGE_WIDTH * RESIZED_IMAGE_HEIGHT))

  for contour in contours:
    # get rectangle bounding contour
    [x, y, w, h] = cv2.boundingRect(contour)

    # crop the letter out of the training image
    cropped_letter = cv_im[y:y+h, x:x+w]

    # resize the cropped letter
    cropped_letter_resized = cv2.resize(cropped_letter, (RESIZED_IMAGE_WIDTH, RESIZED_IMAGE_HEIGHT))

    # show the characters to the trainor and wait for any key press
    cv2.imshow("Cropped Letter", cropped_letter)
    cv2.imshow("Cropped Letter - Resized", cropped_letter_resized)

    pressed_key = cv2.waitKey(0)

    # if the escape key (ascii 27) is pressed, terminate the training
    if pressed_key == 27:
      sys.exit(0)
    elif pressed_key in VALID_CHARACTERS:
      # append the matched ascii equivalent of the pressed key to the list
      matched_characters.append(pressed_key)

      matched_image = cropped_letter_resized.reshape((1, RESIZED_IMAGE_WIDTH * RESIZED_IMAGE_HEIGHT))
      matched_image = np.float32(matched_image)

      matched_images = np.append(matched_images, matched_image, 0)

  matched_characters_flattened = np.array(matched_characters, np.float32)
  matched_characters_reshaped = matched_characters_flattened.reshape((matched_characters_flattened.size, 1))

  np.savetxt(os.path.join(data_dst, 'matched_characters.txt'), matched_characters_reshaped)
  np.savetxt(os.path.join(data_dst, 'matched_images.txt'), matched_images)

  # destroy any existing windows
  cv2.destroyAllWindows()

  return

def initialize_knn_knowledge(char_knowledge_src, image_knowlege_src, **kwargs):
  char_knowledge_dtype = kwargs.get('char_knowledge_dtype', np.float32)
  image_knowledge_dtype = kwargs.get('image_knowledge_dtype', np.float32)

  if not os.path.exists(char_knowledge_src):
    print(f'Character Knowledge Path: {char_knowledge_src} does not exist')
    sys.exit(1)

  if not os.path.exists(image_knowlege_src):
    print(f'Image Knowledge Path: {image_knowlege_src} does not exist')
    sys.exit(1)

  matched_characters = np.loadtxt(char_knowledge_src, char_knowledge_dtype)
  matched_images = np.loadtxt(image_knowlege_src, image_knowledge_dtype)

  matched_characters = matched_characters.reshape((matched_characters.size, 1))

  knn = cv2.ml.KNearest_create()

  knn.train(matched_images, cv2.ml.ROW_SAMPLE, matched_characters)

  return knn

def detect_characters_by_knn(src, dst, knn):
  # initialize string to hold all detected characters
  detected_character_str = ""

  _, contours, _ = cv2.findContours(src, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

  for contour in contours:
    # get bounding rectangle of the detected contour
    [x, y, w, h] = cv2.boundingRect(contour)

    # crop the letter
    cropped_letter = src[y:y+h, x:x+w]

    cropped_letter_resized = cv2.resize(cropped_letter, (RESIZED_IMAGE_WIDTH, RESIZED_IMAGE_HEIGHT))

    matched_image = cropped_letter_resized.reshape((1, RESIZED_IMAGE_WIDTH * RESIZED_IMAGE_HEIGHT))

    matched_image = np.float32(matched_image)

    # returns the (return_value, results, neighbors, distance). we need only the results and discard other values
    _, results, _, _ = knn.findNearest(matched_image, k=1)

    detected_char = str(chr(int(results[0][0])))

    # proceed to the next iteration when no character is detected
    if detected_char is None:
      continue

    # draw rectangle around the contour
    cv2.rectangle(dst, (x, y), (x + w, y + h), (65, 203, 62), 2)

    # append the detected character to the string of detected characters
    detected_character_str = detected_character_str + detected_char

  return detected_character_str


