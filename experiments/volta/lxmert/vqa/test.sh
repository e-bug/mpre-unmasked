#!/bin/bash

source activate /gs/hs0/tgb-deepmt/bugliarello.e/envs/volta

cd ../../../../code/volta
python eval_task.py \
	--bert_model bert-base-uncased --config_file config/lxmert.json --tasks_config_file config_tasks/lxmert_test_tasks.yml \
	--from_pretrained /gs/hs0/tgb-deepmt/bugliarello.e/checkpoints/mpre-unmasked/vqa/volta/lxmert/VQA_lxmert/pytorch_model_19.bin \
	--task 1 \
	--output_dir /gs/hs0/tgb-deepmt/bugliarello.e/results/mpre-unmasked/vqa/volta/lxmert

conda deactivate
