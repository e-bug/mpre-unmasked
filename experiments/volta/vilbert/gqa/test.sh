#!/bin/bash

source activate /gs/hs0/tgb-deepmt/bugliarello.e/envs/volta

cd ../../../../code/volta
python eval_task.py \
        --bert_model bert-base-uncased --config_file config/vilbert_base.json --tasks_config_file config_tasks/vilbert_test_tasks.yml \
        --from_pretrained /gs/hs0/tgb-deepmt/bugliarello.e/checkpoints/mpre-unmasked/gqa/volta/vilbert/GQA_vilbert_base/pytorch_model_19.bin \
        --task 15 \
        --output_dir /gs/hs0/tgb-deepmt/bugliarello.e/results/mpre-unmasked/gqa/volta/vilbert

python score_gqa.py \
        --preds_file /gs/hs0/tgb-deepmt/bugliarello.e/results/mpre-unmasked/gqa/volta/vilbert/pytorch_model_19.bin-/test_result.json \
        --truth_file /gs/hs0/tgb-deepmt/bugliarello.e/data/gqa/testdev_balanced_questions.json

conda deactivate
