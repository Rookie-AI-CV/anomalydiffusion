#!/bin/bash
gpu_id=0
path_to_mvtec="/root/datasets/mvtec"
path_to_the_generated_data="/root/datasets/mvtec_generated_dataset"

echo "training localization"

python train-localization.py --generated_data_path $path_to_the_generated_data  --mvtec_path=$path_to_mvtec

echo "testing localization"

python test-localization.py --generated_data_path $path_to_mvtec