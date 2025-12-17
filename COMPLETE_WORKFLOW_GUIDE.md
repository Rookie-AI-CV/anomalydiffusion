# 完整训练到生成流程指南

本文档提供从数据准备到异常图像生成的完整工作流程。

## 📋 目录

1. [流程概述](#流程概述)
2. [环境准备](#环境准备)
3. [数据准备](#数据准备)
4. [步骤1: 训练异常生成模型](#步骤1-训练异常生成模型)
5. [步骤2: 训练掩码生成模型](#步骤2-训练掩码生成模型)
6. [步骤3: 生成掩码](#步骤3-生成掩码)
7. [步骤4: 生成异常图像](#步骤4-生成异常图像)
8. [完整示例](#完整示例)
9. [输出路径管理](#输出路径管理)
10. [常见问题](#常见问题)

---

## 流程概述

完整的异常生成流程包含以下4个主要步骤：

```
数据准备
    ↓
步骤1: 训练异常生成模型 (main.py)
    ↓
步骤2: 训练掩码生成模型 (train_mask.py)
    ↓
步骤3: 生成掩码 (generate_mask.py)
    ↓
步骤4: 生成异常图像 (generate_with_mask.py)
    ↓
完成！获得异常图像数据集
```

### 输出文件结构

```
项目根目录/
├── logs/                                    # 训练日志（默认）
│   ├── anomaly-checkpoints/                # 异常生成模型
│   │   └── checkpoints/
│   │       ├── spatial_encoder.pt
│   │       └── embeddings.pt
│   └── mask-checkpoints/                   # 掩码生成模型
│       └── {sample_name}-{anomaly_name}/
│           └── checkpoints/
│               └── embeddings.pt
├── generated_mask/                         # 生成的掩码（默认）
│   └── {sample_name}/
│       └── {anomaly_name}/
│           └── *.jpg
└── generated_dataset/                      # 生成的异常图像（默认）
    └── {sample_name}/
        └── {anomaly_name}/
            ├── image/                      # 异常图像
            ├── mask/                       # 掩码
            ├── image-mask/                  # 图像+掩码组合
            ├── ori/                        # 原始图像
            └── recon/                      # 重建图像
```

---

## 环境准备

### 1. 安装依赖

```bash
# 安装Python依赖
pip install -r requirements.txt

# 或使用conda环境
conda env create -f environment.yaml
conda activate anomalydiffusion
```

### 2. 下载基础模型

```bash
# 下载预训练的基础模型
mkdir -p models/ldm/text2img-large/
wget -O models/ldm/text2img-large/model.ckpt \
    https://ommer-lab.com/files/latent-diffusion/nitro/txt2img-f8-large/model.ckpt
```

### 3. 准备数据

确保数据目录结构符合MVTec格式：

```
data_root/
└── {sample_name}/
    ├── train/
    │   └── good/
    │       └── *.jpg
    └── test/
        ├── good/
        │   └── *.jpg
        └── {anomaly_name}/
            └── *.jpg
```

---

## 数据准备

### 创建 name-anomaly.txt

创建 `name-anomaly.txt` 文件，每行一个样本-异常对：

```
gear+crack
gear+chipping
screw+thread_side
wood+color
```

格式：`{sample_name}+{anomaly_name}`

---

## 步骤1: 训练异常生成模型

这一步训练模型学习如何生成异常图像。

### 使用脚本

```bash
bash train_gen_anomaly.sh
```

### 手动执行

```bash
CUDA_VISIBLE_DEVICES=0 python main.py \
    --spatial_encoder_embedding \
    --data_enhance \
    --base configs/latent-diffusion/txt2img-1p4B-finetune-encoder+embedding.yaml \
    -t \
    --actual_resume models/ldm/text2img-large/model.ckpt \
    -n test \
    --gpus 0, \
    --init_word anomaly \
    --mvtec_path=/path/to/mvtec/dataset \
    --logdir /path/to/output/logs
```

### 参数说明

- `--mvtec_path`: MVTec数据集路径
- `--logdir`: 训练输出目录（默认: `logs`）
- `--init_word`: 初始词，通常使用 `anomaly`
- `--spatial_encoder_embedding`: 启用空间编码器和嵌入
- `--data_enhance`: 启用数据增强

### 输出文件

训练完成后，在 `{logdir}/anomaly-checkpoints/checkpoints/` 目录下会生成：
- `spatial_encoder.pt`: 空间编码器权重
- `embeddings.pt`: 嵌入权重

### 自定义输出路径

```bash
# 使用 --logdir 参数指定输出路径
CUDA_VISIBLE_DEVICES=0 python main.py \
    ... \
    --logdir /custom/path/to/outputs
```

---

## 步骤2: 训练掩码生成模型

这一步为每个样本-异常对训练掩码生成模型。

### 使用脚本

编辑 `train_and_generate_mask.sh`，设置参数后运行：

```bash
bash train_and_generate_mask.sh
```

### 手动执行

```bash
CUDA_VISIBLE_DEVICES=0 python train_mask.py \
    --mvtec_path=/path/to/mvtec/dataset \
    --base configs/latent-diffusion/txt2img-1p4B-finetune.yaml \
    -t \
    --actual_resume ./models/ldm/text2img-large/model.ckpt \
    -n test \
    --gpus 0, \
    --init_word crack \
    --sample_name=gear \
    --anomaly_name=crack \
    --logdir logs
```

### 参数说明

- `--sample_name`: 样本名称（如: gear, screw）
- `--anomaly_name`: 异常类型（如: crack, chipping）
- `--mvtec_path`: MVTec数据集路径
- `--logdir`: 训练输出目录（默认: `logs`）
- `--init_word`: 初始词，通常使用 `crack`

### 输出文件

训练完成后，在 `{logdir}/mask-checkpoints/{sample_name}-{anomaly_name}/checkpoints/` 目录下会生成：
- `embeddings.pt`: 掩码生成模型的嵌入权重

### 自定义输出路径

```bash
# 使用 --output_dir 参数指定完整输出路径
CUDA_VISIBLE_DEVICES=0 python train_mask.py \
    ... \
    --output_dir /custom/path/to/mask-training/gear-crack

# 或使用 --logdir（向后兼容）
CUDA_VISIBLE_DEVICES=0 python train_mask.py \
    ... \
    --logdir /custom/path/to/logs
```

---

## 步骤3: 生成掩码

使用训练好的掩码模型生成掩码图像。

### 使用脚本

`train_and_generate_mask.sh` 会自动执行此步骤。

### 手动执行

```bash
CUDA_VISIBLE_DEVICES=0 python generate_mask.py \
    --data_root=/path/to/mvtec/dataset \
    --sample_name=gear \
    --anomaly_name=crack \
    --config configs/latent-diffusion/txt2img-1p4B-finetune.yaml \
    --actual_resume ./models/ldm/text2img-large/model.ckpt \
    --mask_logdir logs \
    --generated_mask_dir ./generated_mask
```

### 参数说明

- `--data_root`: 数据根目录
- `--sample_name`: 样本名称
- `--anomaly_name`: 异常类型
- `--mask_logdir`: 掩码模型保存目录（用于查找embeddings.pt）
- `--mask_embeddings_ckpt`: 直接指定embeddings.pt路径（优先级更高）
- `--generated_mask_dir`: 掩码保存目录（默认: `./generated_mask`）

### 输出文件

生成的掩码保存在 `{generated_mask_dir}/{sample_name}/{anomaly_name}/` 目录下。

### 自定义输出路径

```bash
# 使用 --output_dir 直接指定输出路径（优先级最高）
CUDA_VISIBLE_DEVICES=0 python generate_mask.py \
    ... \
    --output_dir /custom/path/to/masks/gear-crack

# 或使用 --generated_mask_dir（向后兼容）
CUDA_VISIBLE_DEVICES=0 python generate_mask.py \
    ... \
    --generated_mask_dir /custom/path/to/masks
```

---

## 步骤4: 生成异常图像

使用训练好的异常生成模型和生成的掩码，生成最终的异常图像。

### 使用脚本

编辑 `generate_with_mask.sh`，设置参数后运行：

```bash
bash generate_with_mask.sh
```

### 手动执行

```bash
CUDA_VISIBLE_DEVICES=0 python generate_with_mask.py \
    --data_root=/path/to/mvtec/dataset \
    --sample_name=gear \
    --anomaly_name=crack \
    --config configs/latent-diffusion/txt2img-1p4B-finetune-encoder+embedding.yaml \
    --actual_resume ./models/ldm/text2img-large/model.ckpt \
    --spatial_encoder_ckpt /path/to/anomaly-checkpoints/checkpoints/spatial_encoder.pt \
    --embeddings_ckpt /path/to/anomaly-checkpoints/checkpoints/embeddings.pt
```

### 参数说明

- `--data_root`: 数据根目录
- `--sample_name`: 样本名称
- `--anomaly_name`: 异常类型
- `--spatial_encoder_ckpt`: 异常生成模型的空间编码器路径
- `--embeddings_ckpt`: 异常生成模型的嵌入路径
- `--adaptive_mask`: 启用自适应掩码（可选，用于纹理异常）

### 输出文件

生成的异常图像保存在 `generated_dataset/{sample_name}/{anomaly_name}/` 目录下，包含：
- `image/`: 生成的异常图像
- `mask/`: 掩码图像
- `image-mask/`: 图像和掩码的组合
- `ori/`: 原始正常图像
- `recon/`: 重建图像

### 自定义输出路径

```bash
# 使用 --output_dir 参数指定输出路径
CUDA_VISIBLE_DEVICES=0 python generate_with_mask.py \
    ... \
    --output_dir /custom/path/to/anomaly-images/gear-crack
```

---

## 完整示例

### 示例1: 使用默认路径（快速测试）

```bash
# 1. 训练异常生成模型
CUDA_VISIBLE_DEVICES=0 python main.py \
    --spatial_encoder_embedding --data_enhance \
    --base configs/latent-diffusion/txt2img-1p4B-finetune-encoder+embedding.yaml -t \
    --actual_resume models/ldm/text2img-large/model.ckpt \
    -n test --gpus 0, \
    --init_word anomaly \
    --mvtec_path=/root/autodl-tmp/gear-MVTec \
    --logdir logs

# 2. 训练掩码生成模型并生成掩码
bash train_and_generate_mask.sh

# 3. 生成异常图像
bash generate_with_mask.sh
```

### 示例2: 使用自定义输出路径（生产环境）

```bash
# 设置输出根目录
OUTPUT_ROOT="/root/outputs/anomaly-generation"
SAMPLE_NAME="gear"
ANOMALY_NAME="crack"

# 1. 训练异常生成模型
CUDA_VISIBLE_DEVICES=0 python main.py \
    --spatial_encoder_embedding --data_enhance \
    --base configs/latent-diffusion/txt2img-1p4B-finetune-encoder+embedding.yaml -t \
    --actual_resume models/ldm/text2img-large/model.ckpt \
    -n test --gpus 0, \
    --init_word anomaly \
    --mvtec_path=/root/autodl-tmp/gear-MVTec \
    --logdir $OUTPUT_ROOT/anomaly-training

# 2. 训练掩码生成模型
CUDA_VISIBLE_DEVICES=0 python train_mask.py \
    --mvtec_path=/root/autodl-tmp/gear-MVTec \
    --base configs/latent-diffusion/txt2img-1p4B-finetune.yaml -t \
    --actual_resume ./models/ldm/text2img-large/model.ckpt \
    -n test --gpus 0, \
    --init_word crack \
    --sample_name=$SAMPLE_NAME \
    --anomaly_name=$ANOMALY_NAME \
    --output_dir $OUTPUT_ROOT/mask-training/$SAMPLE_NAME-$ANOMALY_NAME

# 3. 生成掩码
CUDA_VISIBLE_DEVICES=0 python generate_mask.py \
    --data_root=/root/autodl-tmp/gear-MVTec \
    --sample_name=$SAMPLE_NAME \
    --anomaly_name=$ANOMALY_NAME \
    --config configs/latent-diffusion/txt2img-1p4B-finetune.yaml \
    --actual_resume ./models/ldm/text2img-large/model.ckpt \
    --mask_embeddings_ckpt $OUTPUT_ROOT/mask-training/$SAMPLE_NAME-$ANOMALY_NAME/checkpoints/embeddings.pt \
    --output_dir $OUTPUT_ROOT/generated-masks/$SAMPLE_NAME-$ANOMALY_NAME

# 4. 生成异常图像
CUDA_VISIBLE_DEVICES=0 python generate_with_mask.py \
    --data_root=/root/autodl-tmp/gear-MVTec \
    --sample_name=$SAMPLE_NAME \
    --anomaly_name=$ANOMALY_NAME \
    --config configs/latent-diffusion/txt2img-1p4B-finetune-encoder+embedding.yaml \
    --actual_resume ./models/ldm/text2img-large/model.ckpt \
    --spatial_encoder_ckpt $OUTPUT_ROOT/anomaly-training/anomaly-checkpoints/checkpoints/spatial_encoder.pt \
    --embeddings_ckpt $OUTPUT_ROOT/anomaly-training/anomaly-checkpoints/checkpoints/embeddings.pt \
    --output_dir $OUTPUT_ROOT/anomaly-images/$SAMPLE_NAME-$ANOMALY_NAME
```

### 示例3: 使用Shell脚本（推荐）

#### 修改 train_and_generate_mask.sh

```bash
# 在脚本开头设置参数
gpu_id=0
data_root="/root/autodl-tmp/crop-mini-mvtec"
sample_name="gear"
anomaly_name="chipping"

# 使用自定义输出路径
mask_output_dir="/root/outputs/mask-training/gear-chipping"
mask_generate_output_dir="/root/outputs/generated-masks/gear-chipping"
```

#### 修改 generate_with_mask.sh

```bash
# 在脚本开头设置参数
gpu_id=0
data_root="/root/autodl-tmp/crop-mini-mvtec"
sample_name="gear"
anomaly_name="chipping"

# 设置模型路径
task_root="/root/outputs/anomaly-training"
spatial_encoder_ckpt="$task_root/anomaly-checkpoints/checkpoints/spatial_encoder.pt"
embeddings_ckpt="$task_root/anomaly-checkpoints/checkpoints/embeddings.pt"

# 使用自定义输出路径
output_dir="/root/outputs/anomaly-images/gear-chipping"
```

---

## 输出路径管理

### 路径参数优先级

1. **train_mask.py**:
   - `--output_dir` > `--logdir`（如果指定了 `--output_dir`，忽略 `--logdir`）

2. **generate_mask.py**:
   - `--output_dir` > `--generated_mask_dir`（如果指定了 `--output_dir`，忽略 `--generated_mask_dir`）
   - `--mask_embeddings_ckpt` > `--mask_logdir`（如果指定了 `--mask_embeddings_ckpt`，忽略 `--mask_logdir`）

3. **generate_with_mask.py**:
   - `--output_dir`（如果未指定，使用默认路径 `generated_dataset/{sample_name}/{anomaly_name}`）

### 推荐的文件组织方式

```
/outputs/
├── anomaly-training/              # 异常生成模型训练
│   └── anomaly-checkpoints/
│       └── checkpoints/
│           ├── spatial_encoder.pt
│           └── embeddings.pt
├── mask-training/                 # 掩码模型训练
│   ├── gear-crack/
│   │   └── checkpoints/
│   │       └── embeddings.pt
│   └── gear-chipping/
│       └── checkpoints/
│           └── embeddings.pt
├── generated-masks/               # 生成的掩码
│   ├── gear-crack/
│   │   └── *.jpg
│   └── gear-chipping/
│       └── *.jpg
└── anomaly-images/                # 生成的异常图像
    ├── gear-crack/
    │   ├── image/
    │   ├── mask/
    │   └── ...
    └── gear-chipping/
        ├── image/
        ├── mask/
        └── ...
```

---

## 常见问题

### Q1: 训练需要多长时间？

- **异常生成模型**: 通常需要数小时到数天，取决于数据集大小和GPU性能
- **掩码生成模型**: 每个样本-异常对通常需要1-3小时
- **生成掩码**: 通常需要几分钟到半小时
- **生成异常图像**: 通常需要几分钟到半小时

### Q2: 需要多少GPU内存？

- 建议使用至少16GB显存的GPU
- 如果显存不足，可以减小batch size

### Q3: 如何批量处理多个样本-异常对？

可以使用循环脚本：

```bash
#!/bin/bash
pairs=("gear+crack" "gear+chipping" "screw+thread_side")

for pair in "${pairs[@]}"; do
    IFS='+' read -r sample anomaly <<< "$pair"
    echo "处理: $sample - $anomaly"
    
    # 训练掩码模型
    python train_mask.py \
        --sample_name=$sample \
        --anomaly_name=$anomaly \
        --output_dir /outputs/mask-training/$sample-$anomaly \
        ...
    
    # 生成掩码
    python generate_mask.py \
        --sample_name=$sample \
        --anomaly_name=$anomaly \
        --output_dir /outputs/masks/$sample-$anomaly \
        ...
done
```

### Q4: 如何恢复中断的训练？

使用 `--resume` 参数：

```bash
python train_mask.py \
    --resume /path/to/logdir \
    ...
```

### Q5: 生成的图像质量不好怎么办？

- 增加训练步数
- 调整学习率
- 使用数据增强（`--data_enhance`）
- 对于纹理异常，使用 `--adaptive_mask`

### Q6: 如何验证生成结果？

检查输出目录中的文件：
- 掩码应该清晰且位于图像中心区域
- 异常图像应该看起来自然
- 原始图像和重建图像应该相似

---

## 下一步

生成异常图像后，可以：

1. **训练检测模型**: 使用生成的异常图像训练异常检测模型
2. **评估性能**: 使用测试集评估检测性能
3. **调整参数**: 根据结果调整训练参数

---

---

## Shell脚本使用说明

项目提供了多个Shell脚本简化操作流程。

### train_gen_anomaly.sh - 训练异常生成模型

**功能**: 训练异常生成模型（步骤1）

**使用方法**:
```bash
# 编辑脚本设置参数
vim train_gen_anomaly.sh

# 运行脚本
bash train_gen_anomaly.sh
```

**需要修改的参数**:
- `gpu_id`: GPU ID
- `path_to_mvtec_dataset`: MVTec数据集路径
- `output_dir`: 训练输出目录（默认: `logs`）

**输出**: `{output_dir}/anomaly-checkpoints/checkpoints/` 下的模型文件

---

### train_and_generate_mask.sh - 训练掩码模型并生成掩码

**功能**: 训练掩码生成模型并生成掩码（步骤2+3）

**使用方法**:
```bash
# 编辑脚本设置参数
vim train_and_generate_mask.sh

# 运行脚本
bash train_and_generate_mask.sh
```

**需要修改的参数**:
- `gpu_id`: GPU ID
- `data_root`: 数据根目录
- `sample_name`: 样本名称（如: gear）
- `anomaly_name`: 异常类型（如: crack）
- `base_model_ckpt`: 基础模型路径
- `mask_output_dir`: 掩码训练输出路径（可选，默认使用 `mask_logdir`）
- `mask_generate_output_dir`: 掩码生成输出路径（可选，默认使用 `generated_mask_dir`）

**输出**:
- 训练输出: `{mask_output_dir}/checkpoints/embeddings.pt` 或 `logs/mask-checkpoints/{sample_name}-{anomaly_name}/checkpoints/embeddings.pt`
- 生成的掩码: `{mask_generate_output_dir}/` 或 `generated_mask/{sample_name}/{anomaly_name}/`

**示例配置**:
```bash
# 使用默认路径
mask_logdir="logs"
generated_mask_dir="./generated_mask"

# 使用自定义路径
mask_output_dir="/root/outputs/mask-training/gear-crack"
mask_generate_output_dir="/root/outputs/generated-masks/gear-crack"
```

---

### generate_with_mask.sh - 生成异常图像

**功能**: 使用训练好的模型和掩码生成异常图像（步骤4）

**使用方法**:
```bash
# 编辑脚本设置参数
vim generate_with_mask.sh

# 运行脚本
bash generate_with_mask.sh
```

**需要修改的参数**:
- `gpu_id`: GPU ID
- `data_root`: 数据根目录
- `sample_name`: 样本名称
- `anomaly_name`: 异常类型
- `task_root`: 异常生成模型的根目录（用于查找checkpoints）
- `spatial_encoder_ckpt`: 空间编码器路径（或使用 `$task_root/anomaly-checkpoints/checkpoints/spatial_encoder.pt`）
- `embeddings_ckpt`: 嵌入路径（或使用 `$task_root/anomaly-checkpoints/checkpoints/embeddings.pt`）
- `output_dir`: 输出目录（可选，默认: `generated_dataset/{sample_name}/{anomaly_name}/`）
- `adaptive_mask`: 是否使用自适应掩码（纹理异常时设为 `true`）

**输出**: `{output_dir}/` 或 `generated_dataset/{sample_name}/{anomaly_name}/` 下的异常图像

**示例配置**:
```bash
# 设置模型路径
task_root="/root/outputs/anomaly-training"
spatial_encoder_ckpt="$task_root/anomaly-checkpoints/checkpoints/spatial_encoder.pt"
embeddings_ckpt="$task_root/anomaly-checkpoints/checkpoints/embeddings.pt"

# 设置输出路径
output_dir="/root/outputs/anomaly-images/gear-crack"

# 纹理异常时启用
adaptive_mask=true
```

---

### 完整流程示例（使用Shell脚本）

```bash
# 1. 训练异常生成模型
bash train_gen_anomaly.sh

# 2. 训练掩码模型并生成掩码
bash train_and_generate_mask.sh

# 3. 生成异常图像
bash generate_with_mask.sh
```

### 脚本参数优先级

- **train_and_generate_mask.sh**:
  - `mask_output_dir` > `mask_logdir`
  - `mask_generate_output_dir` > `generated_mask_dir`

- **generate_with_mask.sh**:
  - `output_dir`（如果未指定，使用默认路径）

---

## 相关文档

- [输出路径使用说明](OUTPUT_PATH_USAGE.md)
- [README](README.md)
- [自定义数据集指南](CUSTOM_DATASET_GUIDE.md)

---

## 支持

如有问题，请检查：
1. 数据路径是否正确
2. 模型文件是否存在
3. GPU内存是否充足
4. 输出目录是否有写入权限

祝使用愉快！🎉

