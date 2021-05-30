#!/bin/bash

source activate /gs/hs0/tgb-deepmt/bugliarello.e/envs/volta

cd ../../../../code/volta
python train_concap.py \
	--bert_model bert-base-uncased --config_file config/ctrl_lxmert.json \
	--train_batch_size 256 --gradient_accumulation_steps 1 --max_seq_length 38 \
	--learning_rate 1e-4 --adam_epsilon 1e-6 --adam_betas 0.9 0.999 --weight_decay 0.01 --warmup_proportion 0.1 --clip_grad_norm 5.0 \
	--objective 1 \
	--file_path /gs/hs0/tgb-deepmt/bugliarello.e/data/conceptual_captions \
	--output_dir /gs/hs0/tgb-deepmt/bugliarello.e/checkpoints/mpre-unmasked/conceptual_captions/volta/ctrl_lxmert \
        --logdir ../../logs/volta/conceptual_captions \
	--num_train_epochs 10 \
#	--resume_file /gs/hs0/tgb-deepmt/bugliarello.e/checkpoints/mpre-unmasked/conceptual_captions/volta/ctrl_lxmert/ctrl_lxmert/pytorch_ckpt_7.tar

conda deactivate
