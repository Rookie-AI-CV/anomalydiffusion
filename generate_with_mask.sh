#!/bin/bash
# 使用训练好的模型和掩码生成异常图像

# 基础配置
gpu_id=0
data_root="/root/autodl-tmp/crop-mini-mvtec"
sample_name="gear"
anomaly_name="chipping"

# 模型路径
config="configs/latent-diffusion/txt2img-1p4B-finetune-encoder+embedding.yaml"
actual_resume="./models/ldm/text2img-large/model.ckpt"
task_root="/root/autodl-tmp/anomaly-task/job_2/output"
spatial_encoder_ckpt="$task_root/anomaly-checkpoints/checkpoints/spatial_encoder.pt"
embeddings_ckpt="$task_root/anomaly-checkpoints/checkpoints/embeddings.pt"

# 输出路径（可选）
output_dir=""  # 默认: generated_dataset/{sample_name}/{anomaly_name}/

# 其他配置
adaptive_mask=false  # 纹理异常时设为true

generate_cmd="CUDA_VISIBLE_DEVICES=$gpu_id python generate_with_mask.py \
    --data_root $data_root \
    --sample_name $sample_name \
    --anomaly_name $anomaly_name \
    --config $config \
    --actual_resume $actual_resume \
    --spatial_encoder_ckpt $spatial_encoder_ckpt \
    --embeddings_ckpt $embeddings_ckpt"

[ -n "$output_dir" ] && generate_cmd="$generate_cmd --output_dir $output_dir"
[ "$adaptive_mask" = true ] && generate_cmd="$generate_cmd --adaptive_mask"

eval $generate_cmd || exit 1

