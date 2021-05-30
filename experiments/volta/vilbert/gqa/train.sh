#!/bin/bash

source activate /gs/hs0/tgb-deepmt/bugliarello.e/envs/volta

cd ../../../../code/volta
python train_tasks.py \
	--bert_model bert-base-uncased --config_file config/vilbert_base.json --tasks_config_file config_tasks/vilbert_trainval_tasks.yml \
	--from_pretrained /gs/hs0/tgb-deepmt/bugliarello.e/checkpoints/mpre-unmasked/conceptual_captions/volta/vilbert/vilbert_base/pytorch_model_9.bin \
	--task 15 \
	--adam_epsilon 1e-6 --adam_betas 0.9 0.999 --weight_decay 0.01 --warmup_proportion 0.1 --clip_grad_norm 0.0 \
	--output_dir /gs/hs0/tgb-deepmt/bugliarello.e/checkpoints/mpre-unmasked/gqa/volta/vilbert \
        --logdir ../../logs/volta/gqa \
	--resume_file /gs/hs0/tgb-deepmt/bugliarello.e/checkpoints/mpre-unmasked/gqa/volta/vilbert/GQA_vilbert_base/pytorch_ckpt_latest.tar

conda deactivate
