#!/bin/bash
# 训练异常生成模型

gpu_id=0
path_to_mvtec_dataset="/root/autodl-tmp/gear-MVTec"
output_dir="/root/autodl-tmp/anomaly-generation-logs" 

CUDA_VISIBLE_DEVICES=$gpu_id python main.py --spatial_encoder_embedding --data_enhance \
 --base configs/latent-diffusion/txt2img-1p4B-finetune-encoder+embedding.yaml -t  \
 --actual_resume models/ldm/text2img-large/model.ckpt -n test --gpus 0,  \
  --init_word anomaly  --mvtec_path=$path_to_mvtec_dataset --logdir $output_dir 