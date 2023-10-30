#!/bin/bash
now=$(date +"%Y%m%d_%H%M%S")

# modify these augments if you want to try other datasets, splits or methods
# dataset: ['endovis2018']
# method: ['unimatch', 'fixmatch', 'supervised']
# exp: just for specifying the 'save_path'
# split: ['1_2', '1_4', '1_8', '1_16']. 
dataset='endovis2018'
method='evaluation'
exp='base/r101/1_2(tous)'
split='1_8'
func='supervised'

config=configs/${dataset}.yaml
labeled_id_path=splits/$dataset/$split/labeled.txt
unlabeled_id_path=splits/$dataset/$split/unlabeled.txt
save_path=exp/$dataset/$func/$exp/ #$split

mkdir -p $save_path

python $method.py \
    --config=$config --labeled-id-path $labeled_id_path --unlabeled-id-path $unlabeled_id_path \
    --save-path $save_path 