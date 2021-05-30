#!/bin/bash

OUTDIR='/data/bugliarello.e/data/mscoco/mscoco_imgfeat/vl-bert'

python tsv_to_jsons.py

exit

#cd $OUTDIR/valid_obj36-36.clean
#zip -0 ../val_frcnn.zip ./*
cd $OUTDIR/train_obj36-36.clean
zip -0 -r ../train_frcnn.zip .
#find . -name '*.json' -print | zip ../train_frcnn.zip -0

