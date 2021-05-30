#!/bin/bash

source activate /gs/hs0/tgb-deepmt/bugliarello.e/envs/volta

cd ../../../../code/volta
python train_tasks.py \
	--bert_model bert-base-uncased --config_file config/ctrl_lxmert.json --tasks_config_file config_tasks/ctrl_trainval_tasks.yml \
	--from_pretrained /gs/hs0/tgb-deepmt/bugliarello.e/checkpoints/mpre-unmasked/conceptual_captions/volta/ctrl_lxmert/ctrl_lxmert/pytorch_model_9.bin \
	--task 12 --drop_last --seed 73 \
	--adam_epsilon 1e-6 --adam_betas 0.9 0.999 --adam_correct_bias --weight_decay 0.0001 --warmup_proportion 0.1 --clip_grad_norm 1.0 \
	--output_dir /gs/hs0/tgb-deepmt/bugliarello.e/checkpoints/mpre-unmasked/nlvr2_s73/volta/ctrl_lxmert \
        --logdir /gs/hs0/tgb-deepmt/bugliarello.e/logs/volta/nlvr2_s73 \
#	--resume_file /gs/hs0/tgb-deepmt/bugliarello.e/checkpoints/mpre-unmasked/nlvr2/volta/ctrl_lxmert/NLVR2_ctrl_lxmert/pytorch_ckpt_latest.tar

conda deactivate
