#!/bin/bash

source activate /gs/hs0/tgb-deepmt/bugliarello.e/envs/volta

model=vilbert
seed=54

cd ../../../../code/volta
python eval_task.py \
	--bert_model bert-base-uncased --config_file config/ctrl_${model}_base.json --tasks_config_file config_tasks/ctrl_test_tasks.yml \
	--from_pretrained /gs/hs0/tgb-deepmt/bugliarello.e/checkpoints/mpre-unmasked/nlvr2/volta/ctrl_${model}_s${seed}/NLVR2_ctrl_${model}_base/pytorch_model_4.bin \
	--task 12 \
	--output_dir /gs/hs0/tgb-deepmt/bugliarello.e/results/mpre-unmasked/nlvr2/volta/ctrl_${model}_s${seed}

conda deactivate
