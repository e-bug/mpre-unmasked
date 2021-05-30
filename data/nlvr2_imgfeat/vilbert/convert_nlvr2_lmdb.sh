#!/bin/bash

source activate vilbert

python convert_nlvr2_lmdb.py

conda deactivate
