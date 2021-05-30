#!/bin/bash

source activate /gs/hs0/tgb-deepmt/bugliarello.e/envs/volta

cd ../../../../code/volta
python eval_task.py \
	--bert_model bert-base-uncased --config_file config/ctrl_uni_vilbert_base.json --tasks_config_file config_tasks/ctrl_test_tasks.yml \
	--from_pretrained /gs/hs0/tgb-deepmt/bugliarello.e/checkpoints/mpre-unmasked/nlvr2/volta/ctrl_uni_vilbert/NLVR2_ctrl_uni_vilbert_base/pytorch_model_7.bin \
	--task 19 \
	--output_dir /gs/hs0/tgb-deepmt/bugliarello.e/results/mpre-unmasked/nlvr2-contrastset/volta/ctrl_uni_vilbert

conda deactivate
