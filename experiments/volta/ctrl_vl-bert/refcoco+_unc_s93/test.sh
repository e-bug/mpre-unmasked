#!/bin/bash

source activate /gs/hs0/tgb-deepmt/bugliarello.e/envs/volta

cd ../../../../code/volta
python eval_task.py \
        --bert_model bert-base-uncased --config_file config/ctrl_vl-bert_base.json --tasks_config_file config_tasks/ctrl_test_tasks.yml \
        --from_pretrained /gs/hs0/tgb-deepmt/bugliarello.e/checkpoints/mpre-unmasked/refcoco+_unc_s93/volta/ctrl_vl-bert/refcoco+_ctrl_vl-bert_base/pytorch_model_19.bin \
        --task 10 \
        --output_dir /gs/hs0/tgb-deepmt/bugliarello.e/results/mpre-unmasked/refcoco+_unc_s93/volta/ctrl_vl-bert

conda deactivate
