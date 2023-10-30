#!/bin/bash
now=$(date +"%Y%m%d_%H%M%S")

# modify these augments if you want to try other datasets, splits or methods
# dataset: ['endovis2018']
# method: ['unimatch', 'fixmatch', 'supervised']
# exp: just for specifying the 'save_path'
# split: ['1_2', '1_4', '1_8', '1_16']. 
dataset='endovis2018'
method='supervised'
exp='base/ch_iou/r101'
split='1_8'

config=configs/${dataset}.yaml
labeled_id_path=splits/$dataset/unsorted/train.txt #$split/labeled.txt
unlabeled_id_path=splits/$dataset/unsorted/$split/unlabeled.txt
save_path=exp/$dataset/$method/$exp/ #$split

mkdir -p $save_path

#python -m torch.distributed.launch \
#    --nproc_per_node=$1 \
#    --master_addr=localhost \
#    --master_port=$2 \
#    $method.py \
#    --config=$config --labeled-id-path $labeled_id_path --unlabeled-id-path $unlabeled_id_path \
#    --save-path $save_path --port $2 2>&1 | tee $save_path/$now.log


python $method.py \
    --config=$config --labeled-id-path $labeled_id_path --unlabeled-id-path $unlabeled_id_path \
    --save-path $save_path # --port $2 2>&1 | tee $save_path/$now.log