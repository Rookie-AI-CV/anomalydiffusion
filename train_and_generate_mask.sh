#!/bin/bash

# ============================================
# Mask 训练和生成脚本
# 用于训练 mask 生成模型并生成 mask 文件
# ============================================

# ============================================
# 基础配置参数
# ============================================
gpu_id=0
data_root="/root/autodl-tmp/crop-mini-mvtec"
sample_name="gear"
anomaly_name="chipping"

# ============================================
# 模型文件路径配置
# ============================================
# 1. 基础模型检查点（用于训练和生成）
base_model_ckpt="./models/ldm/text2img-large/model.ckpt"

# 2. 配置文件路径
config_file="configs/latent-diffusion/txt2img-1p4B-finetune.yaml"

# 3. 训练输出目录（mask 模型会保存到这里）
# 训练完成后，模型会保存到: ${mask_logdir}/mask-checkpoints/${sample_name}-${anomaly_name}/checkpoints/
mask_logdir="logs"  # 默认是 "logs"，可以改为自定义路径

# 4. 生成的 mask 保存目录
generated_mask_dir="./generated_mask"  # mask 文件会保存到: ${generated_mask_dir}/${sample_name}/${anomaly_name}/

# ============================================
# 训练参数
# ============================================
init_word="crack"  # 初始词，通常使用 "crack"
log_name="test"    # 日志名称

echo "============================================"
echo "Mask 训练和生成流程"
echo "============================================"
echo "样本名称: $sample_name"
echo "异常类型: $anomaly_name"
echo "数据路径: $data_root"
echo "GPU ID: $gpu_id"
echo ""
echo "模型文件路径配置:"
echo "  基础模型: $base_model_ckpt"
echo "  配置文件: $config_file"
echo "  Mask 模型保存目录: $mask_logdir/mask-checkpoints/$sample_name-$anomaly_name/checkpoints/"
echo "  生成的 mask 保存目录: $generated_mask_dir/$sample_name/$anomaly_name/"
echo "============================================"

# 步骤1: 训练 mask 生成模型
echo ""
echo "步骤 1/2: 开始训练 mask 生成模型..."
echo "这可能需要较长时间，请耐心等待..."
echo ""

CUDA_VISIBLE_DEVICES=$gpu_id python train_mask.py \
    --mvtec_path=$data_root \
    --base $config_file -t \
    --actual_resume $base_model_ckpt \
    -n $log_name \
    --gpus $gpu_id, \
    --init_word $init_word \
    --sample_name=$sample_name \
    --anomaly_name=$anomaly_name \
    --logdir $mask_logdir \
    --embedding_manager_ckpt ""

# 检查训练是否成功
if [ $? -ne 0 ]; then
    echo ""
    echo "错误: Mask 模型训练失败！"
    echo "请检查错误信息并重试。"
    exit 1
fi

echo ""
echo "✓ Mask 模型训练完成！"
echo ""

# 步骤2: 生成 mask 文件
echo "步骤 2/2: 开始生成 mask 文件..."
echo ""

CUDA_VISIBLE_DEVICES=$gpu_id python generate_mask.py \
    --data_root=$data_root \
    --sample_name=$sample_name \
    --anomaly_name=$anomaly_name \
    --config $config_file \
    --actual_resume $base_model_ckpt \
    --mask_logdir $mask_logdir \
    --generated_mask_dir $generated_mask_dir

# 检查生成是否成功
if [ $? -ne 0 ]; then
    echo ""
    echo "错误: Mask 生成失败！"
    echo "请检查错误信息并重试。"
    exit 1
fi

echo ""
echo "✓ Mask 生成完成！"
echo ""

# 检查生成的文件
mask_dir="$generated_mask_dir/$sample_name/$anomaly_name"
if [ -d "$mask_dir" ]; then
    mask_count=$(ls -1 "$mask_dir" | wc -l)
    echo "生成的 mask 文件数量: $mask_count"
    echo "Mask 保存路径: $mask_dir"
    echo ""
    echo "============================================"
    echo "完成！现在可以运行 generate_with_mask.sh"
    echo "============================================"
else
    echo "警告: 未找到生成的 mask 目录: $mask_dir"
    echo "请检查生成过程是否有错误。"
    exit 1
fi

