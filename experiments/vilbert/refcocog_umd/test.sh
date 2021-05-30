#!/bin/bash

source activate /gs/hs0/tgb-deepmt/bugliarello.e/envs/vilbert-mt

cd ../..
python eval_tasks.py \
	--bert_model bert-base-uncased --config_file config/bert_base_6layer_6conect.json \
	--from_pretrained /gs/hs0/tgb-deepmt/bugliarello.e/checkpoints/refcocog_umd/vilbert/refcocog_bert_base_6layer_6conect/pytorch_model_4.bin \
	--tasks 11 \
	--output_dir /gs/hs0/tgb-deepmt/bugliarello.e/results/refcocog_umd/vilbert

conda deactivate
