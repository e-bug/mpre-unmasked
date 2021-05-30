#!/bin/bash

source activate vilbert

python conceptual_caption_preprocess_sequential_train.py

conda deactivate
