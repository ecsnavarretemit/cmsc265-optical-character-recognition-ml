#!/usr/bin/env python

# train.py
#
# Copyright(c) Exequiel Ceasar Navarrete <esnavarrete1@up.edu.ph>
# Licensed under MIT
# Version 1.0.0-alpha2

import os
from app import create_cv_im_instances_from_dir
from app.ocr import create_knowledgebase

image_instances = create_cv_im_instances_from_dir(os.path.join(os.getcwd(), "assets/img/training"))

create_knowledgebase(image_instances, os.path.join(os.getcwd(), 'data/ocr'))

print("Training complete.")


