#!/bin/bash

source activate vilbert

mkdir -p /data/bugliarello.e/data/refer/refcoco_unc_gt36_imgfeat/vilbert

python convert_refcoco_unc_gt36_lmdb.py

conda deactivate
