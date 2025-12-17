#!/bin/bash
# 训练掩码生成模型并生成掩码

# 基础配置
gpu_id=0
data_root="/root/autodl-tmp/crop-mini-mvtec"
sample_name="gear"
anomaly_name="chipping"

# 模型配置
base_model_ckpt="./models/ldm/text2img-large/model.ckpt"
config_file="configs/latent-diffusion/txt2img-1p4B-finetune.yaml"

# 输出路径（二选一）
mask_logdir="logs"  # 默认路径: logs/mask-checkpoints/{sample_name}-{anomaly_name}/
mask_output_dir=""  # 自定义路径（优先级更高）

generated_mask_dir="./generated_mask"  # 默认路径: generated_mask/{sample_name}/{anomaly_name}/
mask_generate_output_dir=""  # 自定义路径（优先级更高）

# 训练参数
init_word="crack"
log_name="test"

echo "训练掩码模型: $sample_name-$anomaly_name"

# 步骤1: 训练掩码模型

train_cmd="CUDA_VISIBLE_DEVICES=$gpu_id python train_mask.py \
    --mvtec_path=$data_root \
    --base $config_file -t \
    --actual_resume $base_model_ckpt \
    -n $log_name \
    --gpus $gpu_id, \
    --init_word $init_word \
    --sample_name=$sample_name \
    --anomaly_name=$anomaly_name \
    --embedding_manager_ckpt \"\""

[ -n "$mask_output_dir" ] && train_cmd="$train_cmd --output_dir $mask_output_dir" || train_cmd="$train_cmd --logdir $mask_logdir"
eval $train_cmd || exit 1

# 步骤2: 生成掩码

generate_cmd="CUDA_VISIBLE_DEVICES=$gpu_id python generate_mask.py \
    --data_root=$data_root \
    --sample_name=$sample_name \
    --anomaly_name=$anomaly_name \
    --config $config_file \
    --actual_resume $base_model_ckpt"

[ -n "$mask_output_dir" ] && \
    generate_cmd="$generate_cmd --mask_embeddings_ckpt $mask_output_dir/checkpoints/embeddings.pt" || \
    generate_cmd="$generate_cmd --mask_logdir $mask_logdir"

[ -n "$mask_generate_output_dir" ] && \
    generate_cmd="$generate_cmd --output_dir $mask_generate_output_dir" || \
    generate_cmd="$generate_cmd --generated_mask_dir $generated_mask_dir"

eval $generate_cmd || exit 1

mask_dir="${mask_generate_output_dir:-$generated_mask_dir/$sample_name/$anomaly_name}"
[ -d "$mask_dir" ] && echo "完成: $mask_dir" || exit 1

