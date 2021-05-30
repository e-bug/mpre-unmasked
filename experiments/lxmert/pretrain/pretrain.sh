# The name of experiment
name=lxmert

# Create dirs and make backup
output=/data/bugliarello.e/checkpoints/conceptual_captions/$name #/gs/hs0/tgb-deepmt/bugliarello.e/checkpoints/conceptual_captions/$name
mkdir -p $output/src
cp -r src/* $output/src/
cp $0 $output/run.bash

export PYTORCH_PRETRAINED_BERT_CACHE=$output

source activate lxmert

# Pre-training
cd ../../../code/lxmert
CUDA_VISIBLE_DEVICES=$1 PYTHONPATH=$PYTHONPATH:./src \
    python src/pretrain/lxmert_pretrain.py \
    --taskMaskLM --taskObjPredict --taskMatched \
    --visualLosses obj,attr,feat \
    --wordMaskRate 0.15 --objMaskRate 0.15 \
    --train concap_train --valid concap_valid \
    --llayers 9 --xlayers 5 --rlayers 5 \
    --batchSize 256 --optim bert --lr 1e-4 --epochs 20 \
    --tqdm --output $output ${@:2} --multiGPU

conda deactivate
