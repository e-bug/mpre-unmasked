#!/bin/bash

source activate vilbert

mkdir -p /data/bugliarello.e/data/refer/refcoco+_unc_imgfeat/vilbert

python convert_refcoco+_unc_lmdb.py

conda deactivate
