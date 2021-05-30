#!/bin/bash

source activate /gs/hs0/tgb-deepmt/bugliarello.e/envs/vilbert-mt

cd ../..
python train_tasks.py \
	--bert_model bert-base-uncased --config_file config/bert_base_6layer_6conect.json \
	--from_pretrained /gs/hs0/tgb-deepmt/bugliarello.e/checkpoints/conceptual_captions/vilbert/bert_base_6layer_6conect/pytorch_model_9.bin \
	--tasks 15 \
	--lr_scheduler 'warmup_linear' --train_iter_gap 4 \
	--output_dir /gs/hs0/tgb-deepmt/bugliarello.e/checkpoints/gqa/vilbert \
	--resume_file /gs/hs0/tgb-deepmt/bugliarello.e/checkpoints/gqa/vilbert/GQA_bert_base_6layer_6conect/pytorch_ckpt_latest.tar

conda deactivate
