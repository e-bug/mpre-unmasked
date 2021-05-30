#!/bin/bash

source activate /gs/hs0/tgb-deepmt/bugliarello.e/envs/vl-bert

cd ../../../code/vl-bert
python ./scripts/launch.py \
	--nproc_per_node 4 \
	refcoco/train_end2end.py \
	--cfg cfgs/refcoco/base_detected_regions_4x16G.yaml \
	--model-dir /gs/hs0/tgb-deepmt/bugliarello.e/checkpoints/refcoco+_unc/vl-bert/

conda deactivate
