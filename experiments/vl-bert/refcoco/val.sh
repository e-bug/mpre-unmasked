#!/bin/bash

source activate /gs/hs0/tgb-deepmt/bugliarello.e/envs/vl-bert

cd ../../../code/vl-bert
for i in $(seq -w 00 18); do
  python refcoco/test.py \
	--split val \
	--cfg cfgs/refcoco/base_detected_regions_4x16G.yaml \
	--ckpt /gs/hs0/tgb-deepmt/bugliarello.e/checkpoints/refcoco+_unc/vl-bert/output/refcoco+/vlbert/base_detected_regions_4x16G/train_train/vl-bert_base_res101_refcoco-00${i}.model \
	--gpus 0 1 2 3 \
	--result-path /gs/hs0/tgb-deepmt/bugliarello.e/results/refcoco+_unc/vl-bert
done

conda deactivate
