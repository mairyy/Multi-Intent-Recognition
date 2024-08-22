#!/usr/bin bash

for dataset in 'MIntRec'
do
    for seed in 1 #3 4 5 6 7 8 9
    do
        for weight in 0.9 #0.5 0.7
        do
            python run.py \
            --dataset $dataset \
            --logger_name 'tmt' \
            --method 'tmt' \
            --data_mode 'multi-class' \
            --train \
            --save_results \
            --seed $seed \
            --gpu_id '0' \
            --video_feats_path 'video_feats.pkl' \
            --audio_feats_path 'audio_feats.pkl' \
            --text_backbone 'bert-base-uncased' \
            --config_file_name 'tmt' \
            --results_file_name 'tmt.csv' \
            --weight_fuse_relation $weight
        done
    done
done