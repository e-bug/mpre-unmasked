#!/bin/bash

source activate /gs/hs0/tgb-deepmt/bugliarello.e/envs/volta

cd ../../../../code/volta
python train_tasks.py \
        --bert_model bert-base-uncased --config_file config/lxmert.json --tasks_config_file config_tasks/lxmert_trainval_tasks.yml \
        --from_pretrained /gs/hs0/tgb-deepmt/bugliarello.e/checkpoints/mpre-unmasked/conceptual_captions/volta/lxmert/lxmert/pytorch_model_19.bin \
        --task 1 \
        --adam_epsilon 1e-6 --adam_betas 0.9 0.999 --adam_correct_bias --weight_decay 0.01 --warmup_proportion 0.1 --clip_grad_norm 5.0 \
        --output_dir /gs/hs0/tgb-deepmt/bugliarello.e/checkpoints/mpre-unmasked/vqa/volta/lxmert \
        --logdir ../../logs/volta/vqa \
	--resume_file /gs/hs0/tgb-deepmt/bugliarello.e/checkpoints/mpre-unmasked/vqa/volta/lxmert/VQA_lxmert/pytorch_ckpt_latest.tar

conda deactivate
