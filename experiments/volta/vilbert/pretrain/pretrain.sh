#!/bin/bash

source activate /gs/hs0/tgb-deepmt/bugliarello.e/envs/volta

cd ../../../../code/volta
python train_concap.py \
	--bert_model bert-base-uncased --config_file config/vilbert_base.json \
	--train_batch_size 512 --gradient_accumulation_steps 2 \
	--learning_rate 1e-4 --adam_epsilon 1e-8 --adam_betas 0.9 0.98 --weight_decay 0.01 --warmup_proportion 0.1 --clip_grad_norm 0.0 \
	--objective 1 \
	--file_path /gs/hs0/tgb-deepmt/bugliarello.e/data/conceptual_captions \
	--output_dir /gs/hs0/tgb-deepmt/bugliarello.e/checkpoints/mpre-unmasked/conceptual_captions/volta/vilbert \
        --logdir ../../logs/volta/conceptual_captions \
	--start_epoch 0 \
	--num_train_epochs 10 \
	#--resume_file /gs/hs0/tgb-deepmt/bugliarello.e/checkpoints/mpre-unmasked/conceptual_captions/volta/vilbert/vilbert_base/pytorch_ckpt_6.tar \

#         --resume_file /gs/hs0/tgb-deepmt/bugliarello.e/checkpoints/mpre-unmasked/conceptual_captions/vilbert/bert_base_6layer_6conect/pytorch_ckpt_16.tar \
conda deactivate
