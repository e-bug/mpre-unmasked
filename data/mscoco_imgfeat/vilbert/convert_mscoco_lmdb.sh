#!/bin/bash

source activate vilbert

python convert_mscoco_lmdb.py --split trainval
python convert_mscoco_lmdb.py --split test

conda deactivate
