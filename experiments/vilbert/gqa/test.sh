#!/bin/bash

source activate /gs/hs0/tgb-deepmt/bugliarello.e/envs/vilbert-mt

cd ../..
python eval_tasks.py \
        --bert_model bert-base-uncased --config_file config/bert_base_6layer_6conect.json \
        --from_pretrained /gs/hs0/tgb-deepmt/bugliarello.e/checkpoints/gqa/vilbert/GQA_bert_base_6layer_6conect/pytorch_model_19.bin \
        --tasks 15 \
        --output_dir /gs/hs0/tgb-deepmt/bugliarello.e/results/gqa/vilbert

python score_gqa.py \
        --preds_file /gs/hs0/tgb-deepmt/bugliarello.e/results/gqa/vilbert/pytorch_model_19.bin-/test_result.json \
        --truth_file /gs/hs0/tgb-deepmt/bugliarello.e/data/gqa/testdev_balanced_questions.json

conda deactivate
