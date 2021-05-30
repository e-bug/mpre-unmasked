#!/bin/bash

source activate /gs/hs0/tgb-deepmt/bugliarello.e/envs/vilbert-mt

cd ../..
python train_tasks.py \
	--bert_model bert-base-uncased --config_file config/vilbert_base.json \
	--from_pretrained /gs/hs0/tgb-deepmt/bugliarello.e/checkpoints/mpre-unmasked/conceptual_captions/vilbert/bert_base_6layer_6conect/pytorch_model_9.bin \
	--task 12 \
	--lr_scheduler 'warmup_linear' --train_iter_gap 4 --task_specific_tokens \
	--output_dir /gs/hs0/tgb-deepmt/bugliarello.e/checkpoints/mpre-unmasked/nlvr2/vilbert

conda deactivate
