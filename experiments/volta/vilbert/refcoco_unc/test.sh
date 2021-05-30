#!/bin/bash

source activate /gs/hs0/tgb-deepmt/bugliarello.e/envs/volta

cd ../../../../code/volta
python eval_task.py \
        --bert_model bert-base-uncased --config_file config/vilbert_base.json --tasks_config_file config_tasks/vilbert_test_tasks.yml \
        --from_pretrained /gs/hs0/tgb-deepmt/bugliarello.e/checkpoints/mpre-unmasked/refcoco_unc/volta/vilbert/refcoco_vilbert_base/pytorch_model_18.bin \
        --task 9 \
        --output_dir /gs/hs0/tgb-deepmt/bugliarello.e/results/mpre-unmasked/refcoco_unc/volta/vilbert

conda deactivate