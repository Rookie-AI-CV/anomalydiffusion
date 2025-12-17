#!/bin/bash
gpu_id=0
path_to_mvtec="/root/datasets/mvtec"
path_to_the_generated_data="/root/datasets/mvtec_generated_dataset"

echo "training classification"

python train-classification.py --mvtec_path=$path_to_mvtec --generated_data_path=$path_to_the_generated_data

echo "testing classification"

python test-classification.py --mvtec_path=$path_to_mvtec --generated_data_path=$path_to_the_generated_data