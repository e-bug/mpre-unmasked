#!/bin/bash

source activate /gs/hs0/tgb-deepmt/bugliarello.e/envs/vilbert-mt

cd ../..
python eval_tasks.py \
	--bert_model bert-base-uncased --config_file config/bert_base_6layer_6conect.json \
	--from_pretrained /gs/hs0/tgb-deepmt/bugliarello.e/checkpoints/nlvr2/vilbert/NLVR2_bert_base_6layer_6conect/pytorch_model_16.bin \
	--tasks 12 \
	--output_dir /gs/hs0/tgb-deepmt/bugliarello.e/results/nlvr2/vilbert

conda deactivate
