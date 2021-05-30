# The name of this experiment.
name=lxmert

# Save logs and models; make backup.
output=snap/nlvr2/$name
mkdir -p $output/src
cp -r src/* $output/src/
cp $0 $output/run.bash

# See Readme.md for option details.
CUDA_VISIBLE_DEVICES=$1 PYTHONPATH=$PYTHONPATH:./src \
    python src/tasks/nlvr2.py \
    --tiny --llayers 9 --xlayers 5 --rlayers 5 \
    --tqdm --output $output ${@:3}
