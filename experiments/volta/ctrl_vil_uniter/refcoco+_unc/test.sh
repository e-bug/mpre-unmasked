#!/bin/bash

source activate /gs/hs0/tgb-deepmt/bugliarello.e/envs/volta

cd ../../../../code/volta
python eval_task.py \
        --bert_model bert-base-uncased --config_file config/ctrl_vil_uniter_base.json --tasks_config_file config_tasks/ctrl_test_tasks.yml \
        --from_pretrained /gs/hs0/tgb-deepmt/bugliarello.e/checkpoints/mpre-unmasked/refcoco+_unc/volta/ctrl_vil_uniter/refcoco+_ctrl_vil_uniter_base/pytorch_model_5.bin \
        --task 10 \
        --output_dir /gs/hs0/tgb-deepmt/bugliarello.e/results/mpre-unmasked/refcoco+_unc/volta/ctrl_vil_uniter

conda deactivate