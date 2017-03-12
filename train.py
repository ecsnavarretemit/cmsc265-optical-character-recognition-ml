#!/usr/bin/env python

# train.py
#
# Copyright(c) Exequiel Ceasar Navarrete <esnavarrete1@up.edu.ph>
# Licensed under MIT
# Version 1.0.0-alpha1

import os
from app.ocr import create_knowledgebase

create_knowledgebase(os.path.join(os.getcwd(), "assets/img/training"), os.path.join(os.getcwd(), 'data/ocr'))

print("Training complete.")


