# The name of this experiment.
name=lxmert

# Save logs and models; make backup.
output=/gs/hs0/tgb-deepmt/bugliarello.e/checkpoints/vqa/$name
mkdir -p $output/src
cp -r src/* $output/src/
cp $0 $output/run.bash

export PYTORCH_PRETRAINED_BERT_CACHE=$output

source activate /gs/hs0/tgb-deepmt/bugliarello.e/envs/lxmert

# See Readme.md for option details.
CUDA_VISIBLE_DEVICES=$1 PYTHONPATH=$PYTHONPATH:./src \
    python src/tasks/vqa.py \
    --train train --valid valid  \
    --llayers 9 --xlayers 5 --rlayers 5 \
    --loadLXMERT /gs/hs0/tgb-deepmt/bugliarello.e/checkpoints/conceptual_captions/$name/Epoch20 \
    --batchSize 32 --optim bert --lr 5e-5 --epochs 20 \
    --tqdm --output $output ${@:3} \
    --startEpoch 8 --load /gs/hs0/tgb-deepmt/bugliarello.e/checkpoints/vqa/lxmert/BEST

conda deactivate
