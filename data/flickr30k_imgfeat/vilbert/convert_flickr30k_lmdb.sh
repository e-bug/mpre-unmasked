#!/bin/bash

source activate vilbert

#python convert_flickr30k_lmdb.py --split trainval
#python convert_flickr30k_lmdb.py --split test

python convert_flickr30k_lmdb.py --split flickr30k

conda deactivate
