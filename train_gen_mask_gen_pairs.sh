#!/bin/bash
gpu_id=0
path_to_mvtec_dataset="/root/autodl-tmp/crop-mini-mvtec"

CUDA_VISIBLE_DEVICES=$gpu_id python run-mvtec.py --data_path=$path_to_mvtec_dataset