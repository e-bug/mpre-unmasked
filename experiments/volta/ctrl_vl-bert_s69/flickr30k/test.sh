#!/bin/bash

source activate /gs/hs0/tgb-deepmt/bugliarello.e/envs/volta

model=vl-bert
seed=69

cd ../../../../code/volta
python eval_retrieval.py \
        --bert_model bert-base-uncased --config_file config/ctrl_${model}_base.json --tasks_config_file config_tasks/ctrl_test_tasks.yml \
        --from_pretrained /gs/hs0/tgb-deepmt/bugliarello.e/checkpoints/mpre-unmasked/flickr30k/volta/ctrl_${model}_s${seed}/RetrievalFlickr30k_ctrl_${model}_base/pytorch_model_13.bin \
        --output_dir /gs/hs0/tgb-deepmt/bugliarello.e/results/mpre-unmasked/flickr30k/volta/ctrl_${model}_s${seed} \
        --task 8 \
        --split test \
        --batch_size 1

conda deactivate
