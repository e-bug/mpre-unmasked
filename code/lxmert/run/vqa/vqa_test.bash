# The name of this experiment.
name=lxmert

# Save logs and models; make backup.
output=/gs/hs0/tgb-deepmt/bugliarello.e/results/vqa/$name
mkdir -p $output/src
cp -r src/* $output/src/
cp $0 $output/run.bash

export PYTORCH_PRETRAINED_BERT_CACHE=$output

source activate /gs/hs0/tgb-deepmt/bugliarello.e/envs/lxmert

# See Readme.md for option details.
CUDA_VISIBLE_DEVICES=$1 PYTHONPATH=$PYTHONPATH:./src \
    python src/tasks/vqa.py \
    --tiny --train train --valid "" --test test  \
    --llayers 9 --xlayers 5 --rlayers 5 \
    --batchSize 32 --optim bert --lr 5e-5 --epochs 4 \
    --tqdm --output $output ${@:3} \
    --load /gs/hs0/tgb-deepmt/bugliarello.e/checkpoints/vqa/lxmert/BEST

conda deactivate
