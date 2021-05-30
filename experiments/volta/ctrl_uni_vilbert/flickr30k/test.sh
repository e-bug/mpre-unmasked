#!/bin/bash

source activate /gs/hs0/tgb-deepmt/bugliarello.e/envs/volta

cd ../../../../code/volta
python eval_retrieval.py \
        --bert_model bert-base-uncased --config_file config/ctrl_uni_vilbert_base.json --tasks_config_file config_tasks/ctrl_test_tasks.yml \
        --from_pretrained /gs/hs0/tgb-deepmt/bugliarello.e/checkpoints/mpre-unmasked/flickr30k/volta/ctrl_uni_vilbert/RetrievalFlickr30k_ctrl_uni_vilbert_base/pytorch_model_15.bin \
        --output_dir /gs/hs0/tgb-deepmt/bugliarello.e/results/mpre-unmasked/flickr30k/volta/ctrl_uni_vilbert \
        --task 8 \
        --split test \
        --batch_size 1

conda deactivate
