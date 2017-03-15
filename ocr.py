#!/usr/bin/env python

# ocr.py
#
# Copyright(c) Exequiel Ceasar Navarrete <esnavarrete1@up.edu.ph>
# Licensed under MIT
# Version 1.0.0-alpha2

import os
import click
from app import create_cv_im_instances_from_dir
from app.ocr import create_knowledgebase

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

if __name__ == '__main__':
  # add commands to the ocr cli
  ocr.add_command(train)

  ocr()


