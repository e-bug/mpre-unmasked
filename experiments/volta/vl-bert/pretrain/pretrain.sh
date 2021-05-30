#!/bin/bash

source activate /gs/hs0/tgb-deepmt/bugliarello.e/envs/volta

cd ../../../../code/volta
python train_concap.py \
	--bert_model bert-base-uncased --config_file config/vl-bert_base.json \
	--train_batch_size 256 --gradient_accumulation_steps 1 --max_seq_length 25 \
	--learning_rate 256e-7 --adam_epsilon 1e-6 --adam_betas 0.9 0.999 --weight_decay 0.0001 --warmup_steps 8000 --clip_grad_norm 10.0 \
	--objective 2 \
	--file_path /gs/hs0/tgb-deepmt/bugliarello.e/data/conceptual_captions \
	--output_dir /gs/hs0/tgb-deepmt/bugliarello.e/checkpoints/mpre-unmasked/conceptual_captions/volta/vl-bert \
	--logdir ../../logs/volta/conceptual_captions \
	--start_epoch 0 \
	--num_train_epochs 10

#         --resume_file /gs/hs0/tgb-deepmt/bugliarello.e/checkpoints/mpre-unmasked/conceptual_captions/vilbert/bert_base_6layer_6conect/pytorch_ckpt_16.tar \
conda deactivate
