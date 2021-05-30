#!/bin/bash

source activate vilbert

#python extract_captions.py --split train
#python extract_captions.py --split valid
#python extract_captions.py --split test
python extract_captions.py --split test1k

conda deactivate
