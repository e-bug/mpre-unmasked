#!/bin/bash

source activate /gs/hs0/tgb-deepmt/bugliarello.e/envs/vilbert-mt

cd ../..
python eval_retrieval.py \
	--bert_model bert-base-uncased --config_file config/bert_base_6layer_6conect.json \
	--from_pretrained /gs/hs0/tgb-deepmt/bugliarello.e/checkpoints/flickr30k/vilbert/RetrievalFlickr30k_bert_base_6layer_6conect/pytorch_model_18.bin \
	--output_dir /gs/hs0/tgb-deepmt/bugliarello.e/results/flickr30k/vilbert \
	--tasks 8 \
	--split test \
	--batch_size 1 \

conda deactivate
