#!/bin/bash

source activate /gs/hs0/tgb-deepmt/bugliarello.e/envs/vl-bert

cd ../../../code/vl-bert
python refcoco/test.py \
	--split test \
	--cfg cfgs/refcoco/base_detected_regions_4x16G.yaml \
	--ckpt /gs/hs0/tgb-deepmt/bugliarello.e/checkpoints/refcoco+_unc/vl-bert/output/refcoco+/vlbert/base_detected_regions_4x16G/train_train/vl-bert_base_res101_refcoco-0015.model \
	--gpus 0 1 2 3 \
	--result-path /gs/hs0/tgb-deepmt/bugliarello.e/results/refcoco+_unc/vl-bert

conda deactivate
