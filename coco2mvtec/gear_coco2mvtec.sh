#!/bin/bash
# Convert gear dataset from COCO format to MVTec format

set -e  # Exit on error

# Base directory
BASE_DIR="/root/autodl-tmp/gear_dataset-1024"

# Output directory (root directory, category_prefix will be appended)
OUTPUT_DIR="/root/autodl-tmp/gear-MVTec"
CATEGORY_PREFIX="gear"  # 图片类别前缀，用于目录结构和name-anomaly.txt

# Script directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Dataset directories
declare -a DATASET_DIRS=(
    "${BASE_DIR}/上端面-裁剪-新_V18/train"
    "${BASE_DIR}/训练集-背景-新_V30/train"
    "${BASE_DIR}/训练集-新_V12/train"
    "${BASE_DIR}/验证集-新_V5/valid"
    "${BASE_DIR}/验证集-背景-新_V3/valid"
)

# Parameters
MAX_NORMAL_TRAIN=500    # 每类最大正常样本数
MAX_NORMAL_TEST=200    # 每类最大正常测试样本数
MAX_DEFECT_SAMPLES=100  # 每类最大缺陷样本数
EXCLUDE_CATEGORIES=("其他") # 排除类别（当INCLUDE_CATEGORIES为空时生效）
INCLUDE_CATEGORIES=("裂纹")  # 包含类别（为空时选择所有类别，不为空时只选择这些类别）
NAME_MAPPING_FILE="${SCRIPT_DIR}/gear_name_mapping.json"  # 类别名称映射文件

# Crop parameters (set to 0 to disable cropping)
CROP_K_MULTIPLIER=4     # 裁剪倍数K，裁剪尺寸 = max(短边*K, 长边, N)，设置为0禁用裁剪
CROP_MIN_SIZE=256         # 最小裁剪尺寸N（像素），设置为0禁用裁剪

# Normal image crop and defect filter parameters
NORMAL_CROP_SIZE=256      # 正常图片裁剪尺寸（像素），设置为0禁用裁剪
MAX_DEFECT_LONG_SIDE=256    # 缺陷最大长边（像素），设置为0禁用过滤

# Check if script exists
if [ ! -f "${SCRIPT_DIR}/coco2mvtec.py" ]; then
    echo "Error: coco2mvtec.py not found at ${SCRIPT_DIR}/coco2mvtec.py"
    exit 1
fi

# Check if dataset directories exist
for dir in "${DATASET_DIRS[@]}"; do
    if [ ! -d "$dir" ]; then
        echo "Warning: Directory does not exist: $dir"
    fi
done

# Run conversion
echo "Starting COCO to MVTec conversion..."
echo "Output root directory: ${OUTPUT_DIR}"
if [ -n "${CATEGORY_PREFIX}" ]; then
    echo "Category prefix: ${CATEGORY_PREFIX}"
    echo "Actual output directory: ${OUTPUT_DIR}/${CATEGORY_PREFIX}"
fi
echo "Max normal train samples: ${MAX_NORMAL_TRAIN}"
echo "Max normal test samples: ${MAX_NORMAL_TEST}"
echo "Max defect samples per type: ${MAX_DEFECT_SAMPLES}"
if [ ${#INCLUDE_CATEGORIES[@]} -gt 0 ]; then
    echo "Included categories: ${INCLUDE_CATEGORIES[*]}"
else
    echo "Included categories: (all categories)"
    if [ ${#EXCLUDE_CATEGORIES[@]} -gt 0 ]; then
        echo "Excluded categories: ${EXCLUDE_CATEGORIES[*]}"
    fi
fi
# Check if crop is enabled (both parameters must be > 0)
# Use awk for floating point comparison
CROP_ENABLED=$(awk -v k="${CROP_K_MULTIPLIER}" -v n="${CROP_MIN_SIZE}" 'BEGIN {if (k > 0 && n > 0) print "1"; else print "0"}')
if [ "${CROP_ENABLED}" = "1" ]; then
    echo "Defect crop enabled: K=${CROP_K_MULTIPLIER}, Min size=${CROP_MIN_SIZE}"
else
    echo "Defect crop disabled (set CROP_K_MULTIPLIER > 0 and CROP_MIN_SIZE > 0 to enable)"
fi
if [ "${NORMAL_CROP_SIZE}" -gt 0 ]; then
    echo "Normal image crop enabled: size=${NORMAL_CROP_SIZE}x${NORMAL_CROP_SIZE}"
else
    echo "Normal image crop disabled (set NORMAL_CROP_SIZE > 0 to enable)"
fi
if [ "${MAX_DEFECT_LONG_SIDE}" -gt 0 ]; then
    echo "Defect long side filter enabled: max long side=${MAX_DEFECT_LONG_SIDE}"
else
    echo "Defect long side filter disabled (set MAX_DEFECT_LONG_SIDE > 0 to enable)"
fi
echo ""

# Build command arguments
CMD_ARGS=(
    --dirs "${DATASET_DIRS[@]}"
    --output "${OUTPUT_DIR}"
    --max-normal-train "${MAX_NORMAL_TRAIN}"
    --max-normal-test "${MAX_NORMAL_TEST}"
    --max-defect-samples "${MAX_DEFECT_SAMPLES}"
    --name-mapping "${NAME_MAPPING_FILE}"
    --crop-k-multiplier "${CROP_K_MULTIPLIER}"
    --crop-min-size "${CROP_MIN_SIZE}"
    --normal-crop-size "${NORMAL_CROP_SIZE}"
    --max-defect-long-side "${MAX_DEFECT_LONG_SIDE}"
)

# Add include categories if specified
if [ ${#INCLUDE_CATEGORIES[@]} -gt 0 ]; then
    CMD_ARGS+=(--include-categories "${INCLUDE_CATEGORIES[@]}")
else
    # Only add exclude categories if include is not specified
    if [ ${#EXCLUDE_CATEGORIES[@]} -gt 0 ]; then
        CMD_ARGS+=(--exclude-categories "${EXCLUDE_CATEGORIES[@]}")
    fi
fi

# Add category prefix if specified
if [ -n "${CATEGORY_PREFIX}" ]; then
    CMD_ARGS+=(--category-prefix "${CATEGORY_PREFIX}")
fi

python "${SCRIPT_DIR}/coco2mvtec.py" "${CMD_ARGS[@]}"

if [ $? -eq 0 ]; then
    echo ""
    echo "Conversion completed successfully!"
else
    echo ""
    echo "Error: Conversion failed!"
    exit 1
fi
