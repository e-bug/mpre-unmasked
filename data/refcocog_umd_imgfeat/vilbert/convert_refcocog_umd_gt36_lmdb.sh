#!/bin/bash

source activate vilbert

mkdir -p /data/bugliarello.e/data/refer/refcocog_umd_gt_imgfeat/vilbert

python convert_refcocog_umd_gt36_lmdb.py

conda deactivate
