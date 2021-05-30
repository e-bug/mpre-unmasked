#!/bin/bash

source activate /gs/hs0/tgb-deepmt/bugliarello.e/envs/volta

cd ../../../../code/volta
python train_tasks.py \
        --bert_model bert-base-uncased --config_file config/vl-bert_base.json --tasks_config_file config_tasks/vl-bert_trainval_tasks.yml \
        --from_pretrained /gs/hs0/tgb-deepmt/bugliarello.e/checkpoints/mpre-unmasked/conceptual_captions/volta/vl-bert/vl-bert_base/pytorch_model_9.bin \
        --task 10 \
        --adam_epsilon 1e-6 --adam_betas 0.9 0.999 --adam_correct_bias --weight_decay 0.0001 --warmup_steps 500 --clip_grad_norm 1.0 \
        --output_dir /gs/hs0/tgb-deepmt/bugliarello.e/checkpoints/mpre-unmasked/refcoco+_unc/volta/vl-bert \
        --logdir ../../logs/volta/refcoco+_unc

conda deactivate
