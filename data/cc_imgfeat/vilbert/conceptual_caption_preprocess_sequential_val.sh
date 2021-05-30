#!/bin/bash

source activate vilbert

python conceptual_caption_preprocess_sequential_val.py

conda deactivate
