#!/bin/bash

source activate vilbert

mkdir -p /data/bugliarello.e/data/refer/refcoco_unc_imgfeat/vilbert

python convert_refcoco_unc_lmdb.py

conda deactivate
