#!/usr/bin/env python

# ocr.py
#
# Copyright(c) Exequiel Ceasar Navarrete <esnavarrete1@up.edu.ph>
# Licensed under MIT
# Version 1.0.0-alpha2

import os
import cv2
import click
from app import create_cv_im_instances_from_dir, create_binary_image, imclearborder
from app.ocr import create_knowledgebase, count_by_characters, initialize_knn_knowledge, detect_characters_by_knn

@click.group()
def ocr():
  pass

@click.command()
@click.argument('training-images-dir', type=click.Path(exists=True))
@click.argument('output-data-path', type=click.Path(exists=True))
@click.option('--contour-matching-threshold', default=0.25, help='Threshold used in comparing newly detected contours against previously detected contours. The lower the value the better the result will be.')
@click.option('--clear-borders/--no-clear-borders', default=True, help='Clears out borders when this flag is present.')
@click.option('--border-distance', default=50, help='Border distance from the edges.')
@click.option('--extension', default=['jpg'], multiple=True, help='Image file extensions to match (e.g. jpg)')
def train(training_images_dir, output_data_path, contour_matching_threshold, clear_borders, border_distance, extension):
  image_instances = create_cv_im_instances_from_dir(training_images_dir,
                                                    file_exts=list(extension),
                                                    clean_near_border_pixels=clear_borders,
                                                    border_radius=border_distance,
                                                    contour_match_threshold=contour_matching_threshold,
                                                    show_logs=True)

  create_knowledgebase(image_instances, output_data_path)

  click.echo("Training complete")

@click.command()
@click.argument('image', type=click.Path(exists=True))
@click.argument('data-path', type=click.Path(exists=True))
@click.option('--clear-borders/--no-clear-borders', default=True, help='Clears out borders when this flag is present.')
@click.option('--border-distance', default=50, help='Border distance from the edges.')
@click.option('--save-to-file', help='File where the detected characters will be written', type=click.Path())
def recognize(image, data_path, clear_borders, border_distance, save_to_file):
  cv_image = cv2.imread(image)

  # convert to grayscale
  gray_image = cv2.cvtColor(cv_image, cv2.COLOR_BGR2GRAY)

  # convert to binary image
  binary_image = create_binary_image(gray_image)

  # remove borders from the binary image
  if clear_borders is True:
    binary_image = imclearborder(binary_image.copy(), border_distance)

  k_nearest = initialize_knn_knowledge(os.path.join(data_path, "matched_characters.txt"),
                                       os.path.join(data_path, "matched_images.txt"))

  detected_character_str = detect_characters_by_knn(binary_image, cv_image, k_nearest)
  stats = count_by_characters(detected_character_str)

  if save_to_file is None:
    # show the detected string of characters
    print(detected_character_str)

    # show the stats of the characters
    for char, char_count in stats.items():
      print(f"{char}={char_count}")
  else:
    output_file = open(save_to_file, 'w')

    output_file.write(detected_character_str)

    # close the file handle
    output_file.close()

if __name__ == '__main__':
  # add commands to the ocr cli
  ocr.add_command(train)
  ocr.add_command(recognize)

  ocr()


