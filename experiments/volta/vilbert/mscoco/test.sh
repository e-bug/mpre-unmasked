#!/bin/bash

source activate /gs/hs0/tgb-deepmt/bugliarello.e/envs/volta

cd ../../../../code/volta
python eval_retrieval.py \
        --bert_model bert-base-uncased --config_file config/vilbert_base.json --tasks_config_file config_tasks/vilbert_test_tasks.yml \
        --from_pretrained /gs/hs0/tgb-deepmt/bugliarello.e/checkpoints/mpre-unmasked/mscoco/volta/vilbert/RetrievalCOCO_vilbert_base/pytorch_model_18.bin \
        --output_dir /gs/hs0/tgb-deepmt/bugliarello.e/results/mpre-unmasked/mscoco/volta/vilbert \
        --task 7 \
        --split test \
        --batch_size 1

conda deactivate
