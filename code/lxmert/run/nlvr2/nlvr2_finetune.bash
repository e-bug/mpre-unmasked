# The name of this experiment.
name=lxmert

# Save logs and models; Make backup.
output=/gs/hs0/tgb-deepmt/bugliarello.e/checkpoints/nlvr2/$name
mkdir -p $output/src
cp -r src/* $output/src/
cp $0 $output/run.bash

export PYTORCH_PRETRAINED_BERT_CACHE=$output

source activate /gs/hs0/tgb-deepmt/bugliarello.e/envs/lxmert

# See run/Readme.md for option details.
CUDA_VISIBLE_DEVICES=$1 PYTHONPATH=$PYTHONPATH:./src \
    python src/tasks/nlvr2.py \
    --train train --valid valid \
    --llayers 9 --xlayers 5 --rlayers 5 \
    --loadLXMERT /gs/hs0/tgb-deepmt/bugliarello.e/checkpoints/conceptual_captions/$name/Epoch20 \
    --batchSize 32 --optim bert --lr 5e-5 --epochs 4 \
    --tqdm --output $output ${@:3}

conda deactivate
