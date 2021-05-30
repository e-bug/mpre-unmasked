#!/bin/bash

source activate /gs/hs0/tgb-deepmt/bugliarello.e/envs/vl-bert

cd ../../../code/vl-bert
python ./scripts/launch.py \
	--nproc_per_node 4 \
	pretrain/train_end2end.py \
	--cfg cfgs/pretrain/base_prec_withouttextonly_4x16G_fp32.yaml \
	--model-dir /gs/hs0/tgb-deepmt/bugliarello.e/checkpoints/conceptual_captions/vl-bert/

conda deactivate
