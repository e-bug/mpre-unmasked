#!/bin/bash

source activate /gs/hs0/tgb-deepmt/bugliarello.e/envs/volta

cd ../../../../code/volta
python eval_task.py \
	--bert_model bert-base-uncased --config_file config/ctrl_vilbert_base.json --tasks_config_file config_tasks/ctrl_test_tasks.yml \
	--from_pretrained /gs/hs0/tgb-deepmt/bugliarello.e/checkpoints/mpre-unmasked/nlvr2_s73/volta/ctrl_vilbert/NLVR2_ctrl_vilbert_base/pytorch_model_7.bin \
	--task 19 \
	--output_dir /gs/hs0/tgb-deepmt/bugliarello.e/results/mpre-unmasked/nlvr2-contrastset_s73/volta/ctrl_vilbert

conda deactivate
