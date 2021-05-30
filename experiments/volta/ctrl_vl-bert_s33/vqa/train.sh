#!/bin/bash

source activate /gs/hs0/tgb-deepmt/bugliarello.e/envs/volta

model=vl-bert
seed=33

cd ../../../../code/volta
python train_tasks.py \
	--bert_model bert-base-uncased --config_file config/ctrl_${model}_base.json --tasks_config_file config_tasks/ctrl_trainval_tasks.yml \
	--from_pretrained /gs/hs0/tgb-deepmt/bugliarello.e/checkpoints/mpre-unmasked/conceptual_captions_s${seed}/volta/ctrl_${model}/ctrl_${model}_base/pytorch_model_9.bin \
	--task 1 \
	--adam_epsilon 1e-6 --adam_betas 0.9 0.999 --adam_correct_bias --weight_decay 0.0001 --warmup_proportion 0.1 --clip_grad_norm 1.0 \
	--output_dir /gs/hs0/tgb-deepmt/bugliarello.e/checkpoints/mpre-unmasked/vqa/volta/ctrl_${model}_s${seed} \
        --logdir /gs/hs0/tgb-deepmt/bugliarello.e/logs/volta/vqa_s${seed} \

conda deactivate
