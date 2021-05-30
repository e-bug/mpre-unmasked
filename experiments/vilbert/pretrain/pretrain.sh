#!/bin/bash

source activate /gs/hs0/tgb-deepmt/bugliarello.e/envs/vilbert-mt

cd ../..
python train_concap.py \
	--bert_model bert-base-uncased --config_file config/bert_base_6layer_6conect.json \
	--train_batch_size 512 --gradient_accumulation_steps 2 \
	--objective 1 \
	--file_path /gs/hs0/tgb-deepmt/bugliarello.e/data/conceptual_captions \
	--output_dir /gs/hs0/tgb-deepmt/bugliarello.e/checkpoints/conceptual_captions/vilbert \
	--start_epoch 8 \
	--num_train_epochs 10 \
	--resume_file /gs/hs0/tgb-deepmt/bugliarello.e/checkpoints/conceptual_captions/vilbert/bert_base_6layer_6conect/pytorch_ckpt_7.tar

conda deactivate
