#!/bin/bash

source activate /gs/hs0/tgb-deepmt/bugliarello.e/envs/volta

model=visualbert
seed=89

cd ../../../../code/volta
python eval_task.py \
        --bert_model bert-base-uncased --config_file config/ctrl_${model}_base.json --tasks_config_file config_tasks/ctrl_test_tasks.yml \
        --from_pretrained /gs/hs0/tgb-deepmt/bugliarello.e/checkpoints/mpre-unmasked/refcoco+_unc/volta/ctrl_${model}_s${seed}/refcoco+_ctrl_${model}_base/pytorch_model_15.bin \
        --task 10 \
        --output_dir /gs/hs0/tgb-deepmt/bugliarello.e/results/mpre-unmasked/refcoco+_unc/volta/ctrl_${model}_s${seed}

conda deactivate
