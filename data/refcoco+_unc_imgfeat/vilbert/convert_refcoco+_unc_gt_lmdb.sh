#!/bin/bash

source activate vilbert

mkdir -p /data/bugliarello.e/data/refer/refcoco+_unc_gt_imgfeat/vilbert

python convert_refcoco+_unc_gt_lmdb.py

conda deactivate
