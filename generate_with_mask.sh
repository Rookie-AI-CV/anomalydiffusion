#!/bin/bash

# 配置参数
gpu_id=0
data_root="/root/autodl-tmp/crop-mini-mvtec"
sample_name="gear"
anomaly_name="chipping"

# 模型路径配置
config="configs/latent-diffusion/txt2img-1p4B-finetune-encoder+embedding.yaml"
actual_resume="./models/ldm/text2img-large/model.ckpt"
task_root="/root/autodl-tmp/anomaly-task/job_2/output"
spatial_encoder_ckpt="$task_root/anomaly-checkpoints/checkpoints/spatial_encoder.pt"
# 注意：这里应该是异常生成模型的 embeddings，不是 mask 模型的
embeddings_ckpt="$task_root/anomaly-checkpoints/checkpoints/embeddings.pt"

# 是否使用自适应mask（可选）
adaptive_mask=false  # 设置为 true 启用自适应mask

# 运行生成脚本
CUDA_VISIBLE_DEVICES=$gpu_id python generate_with_mask.py \
    --data_root $data_root \
    --sample_name $sample_name \
    --anomaly_name $anomaly_name \
    --config $config \
    --actual_resume $actual_resume \
    --spatial_encoder_ckpt $spatial_encoder_ckpt \
    --embeddings_ckpt $embeddings_ckpt \
    $([ "$adaptive_mask" = true ] && echo "--adaptive_mask")

