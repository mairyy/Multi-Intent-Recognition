#!/usr/bin bash

for dataset in 'MIntRec'
do
    for seed in 1 #2 #4 5 6 7 8 9
    do
        for weight in 0.9 #0.5 0.7
        do
            python run.py \
            --dataset $dataset \
            --logger_name 'a3m' \
            --method 'a3m' \
            --data_mode 'multi-class' \
            --train \
            --save_results \
            --seed $seed \
            --gpu_id '0' \
            --video_feats_path 'video_feats.pkl' \
            --audio_feats_path 'audio_feats.pkl' \
            --text_backbone 'bert-base-uncased' \
            --config_file_name 'a3m' \
            --results_file_name 'a3m.csv' \
            --weight_fuse_relation $weight
        done
    done
done