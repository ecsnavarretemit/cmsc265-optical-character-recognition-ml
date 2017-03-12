# ocr.py
#
# Copyright(c) Exequiel Ceasar Navarrete <esnavarrete1@up.edu.ph>
# Licensed under MIT
# Version 1.0.0-alpha1

import os
import sys
import cv2
import glob
import string
import numpy as np
from app import create_cv_im_instance, create_binary_image, imclearborder, matches_contours

# In ASCII numbers (0-9 and lowercase/uppercase letters)
VALID_CHARACTERS = [ord(char) for char in string.digits + string.ascii_letters]

# Resized image dimensions
RESIZED_IMAGE_WIDTH = 30
RESIZED_IMAGE_HEIGHT = 30

# TODO: add checking if the passed image is a grayscale image
# TODO: replace sys.exists(1) with a custom exception
def create_knowledgebase(data_src_dir, data_dst, **kwargs):
  file_exts = kwargs.get('file_exts', ['jpg', 'png'])
  border_radius = kwargs.get('border_radius', 50)
  show_logs = kwargs.get('show_logs', True)
  contour_match_threshold = kwargs.get('contour_match_threshold', 0.25)

  if not os.path.exists(data_src_dir):
    print(f'Source of Data: {data_src_dir} does not exist')
    sys.exit(1)

  if not os.path.exists(data_dst):
    print(f'Destination of Data: {data_dst} does not exist')
    sys.exit(1)

  # get all images in the directory that matches the extensions provided
  images = []
  for ext in file_exts:
    images.extend(glob.glob(f"{data_src_dir}/*.{ext}"))

  # terminate if no images are found
  if len(images) == 0:
    print(f'No images in the source directory {data_src_dir}')
    sys.exit(1)

  # initialize data containers
  matched_characters = []
  matched_images = np.empty((0, RESIZED_IMAGE_WIDTH * RESIZED_IMAGE_HEIGHT))

  # convert images list to cv image instances list
  cv_img_instances = list(map(create_cv_im_instance, images))

  print("Instruction: Please press the character that corresponds to the character being presented in the window.")

  # Create holder for the previous processed contour, pressed_key for comparison
  contour_match_status = False
  prev_matched_contours = []
  prev_pressed_key = None

  for cv_item in cv_img_instances:
    if show_logs is True:
      print(f"Extracting characters from: {cv_item['path']}")

    # convert to grayscale
    gray_image = cv2.cvtColor(cv_item['cv_im'], cv2.COLOR_BGR2GRAY)

    # convert to binary image
    binary_image = create_binary_image(gray_image)

    # remove borders from the binary image
    binary_image = imclearborder(binary_image.copy(), border_radius)

    # we only need the contours, just discard the first and last elements returned by the function
    _, contours, _ = cv2.findContours(binary_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    for contour in contours:
      if len(prev_matched_contours) > 0:
        contour_match_status = matches_contours(prev_matched_contours, contour, threshold=contour_match_threshold)
        prev_matched_contours.clear()

        print(f"Contour match status: {str(contour_match_status)}")

      # append the first detected contour to the list
      if len(prev_matched_contours) == 0:
        prev_matched_contours.append(contour)

      # get rectangle bounding contour
      [x, y, w, h] = cv2.boundingRect(contour)

      # crop the letter out of the training image
      cropped_letter = binary_image[y:y+h, x:x+w]

      # resize the cropped letter
      cropped_letter_resized = cv2.resize(cropped_letter, (RESIZED_IMAGE_WIDTH, RESIZED_IMAGE_HEIGHT))

      pressed_key = None

      if contour_match_status is True:
        print("Skipping contour since it falls within below the threshold.")
        pressed_key = prev_pressed_key
      else:
        # show the characters to the trainor and wait for any key press
        cv2.imshow("Cropped Letter", cropped_letter)
        cv2.imshow("Cropped Letter - Resized", cropped_letter_resized)

        pressed_key = cv2.waitKey(0)
        prev_pressed_key = pressed_key

      # if the escape key (ascii 27) is pressed, terminate the training
      if pressed_key == 27:
        if show_logs is True:
          print("Cancelling training. Thank you.")

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

def count_by_characters(detected_chars):
  char_dict = {}

  for valid_char in VALID_CHARACTERS:
    resolved_char = str(chr(int(valid_char)))

    char_dict[resolved_char] = detected_chars.count(resolved_char)

  return char_dict


