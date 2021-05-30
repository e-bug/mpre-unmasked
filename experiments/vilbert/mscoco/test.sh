#!/bin/bash

source activate /gs/hs0/tgb-deepmt/bugliarello.e/envs/vilbert-mt

cd ../..
python eval_retrieval.py \
	--bert_model bert-base-uncased --config_file config/bert_base_6layer_6conect.json \
	--from_pretrained /gs/hs0/tgb-deepmt/bugliarello.e/checkpoints/mscoco/vilbert/RetrievalCOCO_bert_base_6layer_6conect/pytorch_model_19.bin \
	--output_dir /gs/hs0/tgb-deepmt/bugliarello.e/results/mscoco/vilbert \
	--tasks 7 \
	--split test \
	--batch_size 1 \

conda deactivate
