#!/usr/bin/env python3
"""
Convert COCO format object detection data to MVTec dataset format
Support merging from multiple directories, mask images generated from bounding boxes (bbox)
"""

import argparse
import json
import shutil
from pathlib import Path
from datetime import datetime
import numpy as np
import cv2
from tqdm import tqdm


def bbox_to_mask(bbox, img_w, img_h):
    x, y, w, h = int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3])
    mask = np.zeros((img_h, img_w), dtype=np.uint8)
    mask[max(0, y):min(y + h, img_h), max(0, x):min(x + w, img_w)] = 255
    return mask


def calculate_crop_region(bboxes, img_w, img_h, k_multiplier, min_size):
    """计算裁剪区域: max(短边*K, 长边, N)，超出边界时调整位置，无法满足则返回None"""
    if not bboxes:
        return None
    
    min_x = min(bbox[0] for bbox in bboxes)
    min_y = min(bbox[1] for bbox in bboxes)
    max_x = max(bbox[0] + bbox[2] for bbox in bboxes)
    max_y = max(bbox[1] + bbox[3] for bbox in bboxes)
    
    defect_w = max_x - min_x
    defect_h = max_y - min_y
    short_side = min(defect_w, defect_h)
    long_side = max(defect_w, defect_h)
    
    crop_w = max(short_side * k_multiplier, long_side, min_size)
    crop_h = max(short_side * k_multiplier, long_side, min_size)
    crop_size = max(crop_w, crop_h)
    crop_w = crop_h = crop_size
    
    if crop_w > img_w or crop_h > img_h:
        return None
    
    defect_center_x = (min_x + max_x) / 2
    defect_center_y = (min_y + max_y) / 2
    crop_x = defect_center_x - crop_w / 2
    crop_y = defect_center_y - crop_h / 2
    
    if crop_x < 0:
        crop_x = 0
    if crop_y < 0:
        crop_y = 0
    if crop_x + crop_w > img_w:
        crop_x = img_w - crop_w
    if crop_y + crop_h > img_h:
        crop_y = img_h - crop_h
    
    if min_x < crop_x or max_x > crop_x + crop_w or min_y < crop_y or max_y > crop_y + crop_h:
        return None
    
    return (int(crop_x), int(crop_y), int(crop_w), int(crop_h))


def random_crop_to_size(img, target_size=256):
    """随机裁剪图像到指定尺寸，如果图像小于目标尺寸则返回None"""
    h, w = img.shape[:2]
    if w < target_size or h < target_size:
        return None
    
    max_x = w - target_size
    max_y = h - target_size
    crop_x = np.random.randint(0, max_x + 1)
    crop_y = np.random.randint(0, max_y + 1)
    
    return img[crop_y:crop_y+target_size, crop_x:crop_x+target_size]




def merge_coco_data(coco_dirs, target_categories=None, name_mapping=None, include_categories=None):
    all_images, all_annotations, categories_dict = [], [], {}
    img_offset, ann_offset = 0, 0
    name_mapping = name_mapping or {}
    
    for coco_dir in tqdm(coco_dirs, desc="Loading COCO files"):
        coco_file = coco_dir / '_annotations.coco.json'
        if not coco_file.exists():
            continue
        
        try:
            data = json.load(open(coco_file, 'r', encoding='utf-8'))
        except Exception as e:
            print(f"Warning: Failed to read {coco_file}: {e}")
            continue
        
        cat_map = {}
        for cat in data.get('categories', []):
            name = cat['name']
            # 如果指定了包含类别，只处理这些类别
            if include_categories and name not in include_categories:
                continue
            # 如果指定了排除类别，跳过这些类别
            if target_categories and name in target_categories:
                continue
            
            if name_mapping and name in name_mapping:
                mapped_name = name_mapping[name]
            else:
                mapped_name = name
            
            if mapped_name not in categories_dict:
                categories_dict[mapped_name] = len(categories_dict) + 1
            cat_map[cat['id']] = categories_dict[mapped_name]
        
        for img in data.get('images', []):
            img_new = img.copy()
            img_new['id'] += img_offset
            img_new['file_name'] = str(coco_dir / img['file_name'])
            all_images.append(img_new)
        
        for ann in data.get('annotations', []):
            if 'bbox' not in ann or ann['category_id'] not in cat_map:
                continue
            ann_new = ann.copy()
            ann_new['id'] += ann_offset
            ann_new['image_id'] += img_offset
            ann_new['category_id'] = cat_map[ann['category_id']]
            all_annotations.append(ann_new)
        
        img_offset += len(data.get('images', []))
        ann_offset += len(data.get('annotations', []))
    
    return all_images, all_annotations, categories_dict


def write_readme(output_dir, coco_dirs, stats, target_categories, train_ratio, 
                background_class, min_samples, max_normal_train, max_normal_test, 
                max_defect_samples, name_mapping_file, crop_k_multiplier, crop_min_size, 
                name_mapping=None, normal_crop_size=0, max_defect_long_side=0):
    """生成README.md"""
    readme_path = output_dir / 'README.md'
    
    # 创建反向映射：从英文名映射回中文名
    reverse_mapping = {}
    if name_mapping:
        reverse_mapping = {v: k for k, v in name_mapping.items()}
    
    defect_types = sorted(stats['defect_types'].keys())
    
    with open(readme_path, 'w', encoding='utf-8') as f:
        f.write("# MVTec数据集说明文档\n\n")
        f.write(f"**生成时间**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        
        f.write("## 数据来源\n\n")
        for i, dir_path in enumerate(coco_dirs, 1):
            f.write(f"{i}. `{dir_path}`\n")
        f.write("\n")
        
        f.write("## 数据集结构\n\n")
        f.write("```\n")
        f.write("train/\n")
        f.write(f"  {background_class}/  # 训练集正常样本\n")
        f.write("test/\n")
        f.write(f"  {background_class}/  # 测试集正常样本\n")
        for dt in defect_types:
            original_name = reverse_mapping.get(dt, dt)
            f.write(f"  {dt}/  # {original_name}的测试样本\n")
        f.write("ground_truth/\n")
        for dt in defect_types:
            original_name = reverse_mapping.get(dt, dt)
            f.write(f"  {dt}/  # {original_name}的掩码图\n")
        f.write("```\n\n")
        
        f.write("## 数据集统计\n\n")
        f.write(f"- **训练集正常样本**: {stats['train_good']} 张\n")
        f.write(f"- **测试集正常样本**: {stats['test_good']} 张\n")
        f.write(f"- **剔除多缺陷类别图片**: {stats['skipped_multi_defect']} 张\n")
        if max_defect_long_side > 0:
            f.write(f"- **因缺陷长边超过{max_defect_long_side}而跳过的图片**: {stats['skipped_defect_too_large']} 张\n")
        if crop_k_multiplier > 0 and crop_min_size > 0:
            f.write(f"- **因裁剪约束而跳过的图片**: {stats['skipped_crop_failed']} 张\n")
        if normal_crop_size > 0:
            f.write(f"- **因正常图片尺寸小于{normal_crop_size}而跳过的图片**: {stats['skipped_normal_too_small']} 张\n")
        f.write(f"- **缺陷类型数量**: {len(stats['defect_types'])} 种\n\n")
        
        f.write("### 缺陷类型详细统计\n\n")
        f.write("| 缺陷类型 | 原始名称 | 样本数 |\n")
        f.write("|---------|---------|--------|\n")
        for dt, c in sorted(stats['defect_types'].items()):
            original_name = reverse_mapping.get(dt, dt)
            f.write(f"| {dt} | {original_name} | {c} 张 |\n")
        f.write("\n")
        
        if stats.get('filtered_types'):
            f.write(f"### 已过滤的类型（样本数 < {min_samples}）\n\n")
            f.write("| 缺陷类型 | 原始名称 | 样本数 |\n")
            f.write("|---------|---------|--------|\n")
            for dt, c in sorted(stats['filtered_types'].items()):
                original_name = reverse_mapping.get(dt, dt)
                f.write(f"| {dt} | {original_name} | {c} 张 |\n")
            f.write("\n")
        
        f.write("## 生成参数\n\n")
        f.write(f"- **训练集比例**: {train_ratio}\n")
        f.write(f"- **背景类别名称**: {background_class}\n")
        if target_categories:
            f.write(f"- **排除的类别**: {', '.join(target_categories)}\n")
        if min_samples > 0:
            f.write(f"- **最小样本数阈值**: {min_samples}\n")
        if max_normal_train > 0:
            f.write(f"- **训练集正常样本最大数量**: {max_normal_train}\n")
        if max_normal_test > 0:
            f.write(f"- **测试集正常样本最大数量**: {max_normal_test}\n")
        if max_defect_samples > 0:
            f.write(f"- **每类缺陷样本最大数量**: {max_defect_samples}\n")
        if name_mapping_file:
            f.write(f"- **名称映射文件**: `{name_mapping_file}`\n")
        if normal_crop_size > 0:
            f.write(f"- **正常样本裁剪尺寸**: {normal_crop_size}×{normal_crop_size} 像素\n")
        if max_defect_long_side > 0:
            f.write(f"- **缺陷最大长边**: {max_defect_long_side} 像素\n")
        if crop_k_multiplier > 0 and crop_min_size > 0:
            f.write(f"- **缺陷样本裁剪功能**: 已启用\n")
            f.write(f"  - **裁剪倍数K**: {crop_k_multiplier}\n")
            f.write(f"  - **最小尺寸N**: {crop_min_size} 像素\n")
            f.write(f"  - **裁剪规则**: `max(短边 × {crop_k_multiplier}, 长边, {crop_min_size})`\n")
        f.write("\n")
        
        f.write("## 处理流程说明\n\n")
        f.write("### 1. 数据加载与合并\n")
        f.write("- 从多个COCO格式数据目录加载数据\n")
        f.write("- 合并所有标注文件和图像\n")
        f.write("- 应用类别名称映射（如提供）\n")
        f.write("- 排除指定类别（如提供）\n\n")
        
        f.write("### 2. 数据筛选\n")
        f.write("按以下顺序进行筛选：\n\n")
        f.write("#### 2.1 剔除多缺陷类别图片\n")
        f.write("- 如果一张图片包含多个不同的缺陷类别，该图片将被剔除\n")
        f.write("- 只保留包含单一缺陷类别的图片\n\n")
        
        if max_defect_long_side > 0:
            f.write("#### 2.2 缺陷长边过滤\n")
            f.write("- 计算所有缺陷框的合并边界框\n")
            f.write("- 计算缺陷的长边（max(宽度, 高度)）\n")
            f.write(f"- 如果缺陷的长边超过{max_defect_long_side}像素，该样本将被跳过\n")
            f.write(f"- 共跳过 {stats['skipped_defect_too_large']} 张缺陷长边超过{max_defect_long_side}的图片\n\n")
        
        if crop_k_multiplier > 0 and crop_min_size > 0:
            section_num = "2.3" if max_defect_long_side > 0 else "2.2"
            f.write(f"#### {section_num} 缺陷样本裁剪约束检查\n")
            f.write("- 对每个缺陷样本检查是否满足裁剪约束\n")
            f.write("- 裁剪尺寸计算: `max(缺陷短边 × K, 缺陷长边, N像素)`\n")
            f.write("- 裁剪框为正方形，以缺陷中心为基准\n")
            f.write("- 如果裁剪框超出图像边界，自动调整位置确保缺陷完整包含\n")
            f.write("- 如果调整后仍无法满足条件（需要填充），该样本将被跳过\n")
            f.write(f"- 共跳过 {stats['skipped_crop_failed']} 张不满足裁剪约束的图片\n\n")
        
        if normal_crop_size > 0:
            section_num = "2.4" if (max_defect_long_side > 0 and crop_k_multiplier > 0 and crop_min_size > 0) else ("2.3" if (max_defect_long_side > 0 or (crop_k_multiplier > 0 and crop_min_size > 0)) else "2.2")
            f.write(f"#### {section_num} 正常样本尺寸检查\n")
            f.write(f"- 如果启用了正常样本裁剪，检查图像尺寸是否满足要求\n")
            f.write(f"- 如果图像尺寸小于{normal_crop_size}×{normal_crop_size}，该样本将被跳过\n")
            f.write(f"- 共跳过 {stats['skipped_normal_too_small']} 张尺寸不足的正常图片\n\n")
        
        section_num = "2.5" if (max_defect_long_side > 0 and crop_k_multiplier > 0 and crop_min_size > 0 and normal_crop_size > 0) else ("2.4" if ((max_defect_long_side > 0 and crop_k_multiplier > 0 and crop_min_size > 0) or normal_crop_size > 0) else ("2.3" if (max_defect_long_side > 0 or (crop_k_multiplier > 0 and crop_min_size > 0) or normal_crop_size > 0) else "2.2"))
        f.write(f"#### {section_num} 样本数量限制（随机采样）\n")
        f.write("- **重要**: 所有过滤和跳过行为都在采样前完成\n")
        f.write("- 在完成上述所有筛选后，对剩余样本进行随机采样\n")
        f.write("- 采样后的样本数量将尽可能接近设定的采样数量\n")
        if max_normal_train > 0:
            f.write(f"- 训练集正常样本: 随机选择最多 {max_normal_train} 张\n")
        if max_normal_test > 0:
            f.write(f"- 测试集正常样本: 随机选择最多 {max_normal_test} 张\n")
        if max_defect_samples > 0:
            f.write(f"- 每类缺陷样本: 随机选择最多 {max_defect_samples} 张\n")
        if max_normal_train == 0 and max_normal_test == 0 and max_defect_samples == 0:
            f.write("- 不限制样本数量，保留所有通过筛选的样本\n")
        f.write("\n")
        
        f.write("### 3. 数据划分\n")
        f.write(f"- 按照训练集比例 {train_ratio} 划分训练集和测试集\n")
        f.write("- 正常样本（无缺陷）: 根据划分结果放入 train 或 test\n")
        f.write("- 缺陷样本: 全部放入 test（MVTec格式要求）\n\n")
        
        f.write("### 4. 图像处理\n")
        if normal_crop_size > 0:
            f.write(f"- **正常样本**: 随机裁剪到{normal_crop_size}×{normal_crop_size}像素\n")
            f.write(f"  - 尺寸检查已在采样前完成，所有采样的样本都能成功裁剪\n")
        else:
            f.write("- **正常样本**: 直接复制原始图像（未启用裁剪）\n")
        if crop_k_multiplier > 0 and crop_min_size > 0:
            f.write("- **缺陷样本**: 根据裁剪约束进行裁剪处理\n")
            f.write("  - 计算裁剪区域并执行裁剪\n")
            f.write("  - 调整标注框坐标到裁剪后的坐标系\n")
            f.write("  - 保存裁剪后的图像\n")
        else:
            f.write("- **缺陷样本**: 直接复制原始图像（未启用裁剪）\n")
        f.write("\n")
        
        f.write("### 5. 掩码生成\n")
        f.write("- 根据目标检测框（bbox）生成掩码图\n")
        f.write("- 掩码图保存为PNG格式，缺陷区域为白色（255），背景为黑色（0）\n")
        if crop_k_multiplier > 0 and crop_min_size > 0:
            f.write("- 如果启用了裁剪，掩码图尺寸与裁剪后的图像一致\n")
        f.write("\n")
        
        f.write("### 6. 最终过滤\n")
        if min_samples > 0:
            f.write(f"- 如果某类缺陷样本数 < {min_samples}，该类将被完全移除\n")
            f.write("- 包括图像文件和掩码文件都会被删除\n")
        else:
            f.write("- 不进行最终过滤\n")
        f.write("\n")
        
        f.write("## 说明\n\n")
        f.write("1. 本数据集由COCO格式转换而来\n")
        f.write("2. 掩码图由目标检测框（bbox）生成\n")
        f.write("3. 同一张图片包含多个缺陷类别的样本已被剔除\n")
        item_num = 3
        if max_defect_long_side > 0:
            item_num += 1
            f.write(f"{item_num}. 缺陷长边超过{max_defect_long_side}像素的样本已被过滤\n")
        if normal_crop_size > 0:
            item_num += 1
            f.write(f"{item_num}. 正常样本随机裁剪到{normal_crop_size}×{normal_crop_size}像素，尺寸检查在采样前完成，采样后的样本都能成功裁剪\n")
        if max_normal_train > 0 or max_normal_test > 0 or max_defect_samples > 0:
            item_num += 1
            f.write(f"{item_num}. 样本数量已通过随机选择进行限制，采样在完成所有过滤后进行，确保最终数量尽可能接近设定值\n")
        if crop_k_multiplier > 0 and crop_min_size > 0:
            item_num += 1
            f.write(f"{item_num}. 缺陷样本裁剪处理说明:\n")
            f.write(f"   - 裁剪尺寸计算: `max(缺陷短边 × {crop_k_multiplier}, 缺陷长边, {crop_min_size}像素)`\n")
            f.write("   - 裁剪框为正方形，以缺陷中心为基准\n")
            f.write("   - 如果裁剪框超出图像边界，会自动调整位置确保缺陷完整包含\n")
            f.write("   - 如果调整后仍无法满足条件（需要填充），该样本将被跳过\n")
            f.write("   - 裁剪后的图像和掩码坐标已相应调整\n")


def coco_to_mvtec(coco_dirs, output_dir, target_categories=None, train_ratio=0.8, 
                  background_class='good', min_samples=0, max_normal_train=0, 
                  max_normal_test=0, max_defect_samples=0, name_mapping_file=None,
                  crop_k_multiplier=0, crop_min_size=0, normal_crop_size=0, max_defect_long_side=0,
                  include_categories=None, category_prefix=None):
    coco_dirs = [Path(d) for d in coco_dirs]
    output_dir = Path(output_dir)
    
    # 保存原始输出目录（用于生成 name-anomaly.txt）
    original_output_dir = output_dir
    
    # 如果指定了类别前缀，输出目录结构为：output_dir/category_prefix/
    if category_prefix:
        output_dir = output_dir / category_prefix
    
    valid_dirs = [d for d in coco_dirs if d.exists() and (d / '_annotations.coco.json').exists()]
    if not valid_dirs:
        print("Error: No valid COCO dataset directories found")
        return
    
    print(f"Processing {len(valid_dirs)} COCO dataset directories")
    
    name_mapping = {}
    if name_mapping_file:
        try:
            with open(name_mapping_file, 'r', encoding='utf-8') as f:
                name_mapping = json.load(f)
            print(f"Loaded {len(name_mapping)} category name mappings from {name_mapping_file}")
        except Exception as e:
            print(f"Warning: Failed to load name mapping file: {e}")
    
    all_images, all_annotations, categories_dict = merge_coco_data(valid_dirs, target_categories, name_mapping, include_categories)
    cat_reverse = {v: k for k, v in categories_dict.items()}
    if include_categories:
        print(f"Included categories: {', '.join(include_categories)}")
    elif target_categories:
        print(f"Excluded categories: {', '.join(target_categories)}")
    else:
        print(f"All categories: {', '.join(categories_dict.keys())}")
    
    images_dict = {img['id']: img for img in all_images}
    ann_by_img = {}
    for ann in all_annotations:
        ann_by_img.setdefault(ann['image_id'], []).append(ann)
    
    train_dir, test_dir, gt_dir = output_dir / 'train' / background_class, output_dir / 'test' / background_class, output_dir / 'ground_truth'
    for d in [train_dir, test_dir, gt_dir]:
        d.mkdir(parents=True, exist_ok=True)
    
    stats = {'train_good': 0, 'test_good': 0, 'defect_types': {}, 'skipped_multi_defect': 0, 
             'filtered_types': {}, 'skipped_crop_failed': 0, 'skipped_defect_too_large': 0, 
             'skipped_normal_too_small': 0}
    defect_files = {}
    
    np.random.seed(42)
    img_list = list(images_dict.values())
    np.random.shuffle(img_list)
    train_set = set(img['id'] for img in img_list[:int(len(img_list) * train_ratio)])
    
    normal_train_candidates = []
    normal_test_candidates = []
    defect_candidates = {}
    
    for img_info in tqdm(img_list, desc="Collecting candidates"):
        img_id, img_path = img_info['id'], Path(img_info['file_name'])
        if not img_path.exists():
            continue
        
        img = cv2.imread(str(img_path))
        if img is None:
            continue
        
        is_train = img_id in train_set
        anns = ann_by_img.get(img_id, [])
        
        if not anns:
            # 如果启用了正常图片裁剪，在收集候选时检查尺寸
            if normal_crop_size > 0:
                h, w = img.shape[:2]
                if w < normal_crop_size or h < normal_crop_size:
                    stats['skipped_normal_too_small'] += 1
                    continue
            
            if is_train:
                normal_train_candidates.append((img_id, img_path))
            else:
                normal_test_candidates.append((img_id, img_path))
            continue
        
        defect_cats = {}
        for ann in anns:
            defect_cats.setdefault(cat_reverse[ann['category_id']], []).append(ann['bbox'])
        
        if len(defect_cats) > 1:
            stats['skipped_multi_defect'] += 1
            continue
        
        defect_type, bboxes = list(defect_cats.items())[0]
        
        # 检查缺陷长边是否超过阈值
        if max_defect_long_side > 0:
            # 计算所有缺陷框的合并边界框
            min_x = min(bbox[0] for bbox in bboxes)
            min_y = min(bbox[1] for bbox in bboxes)
            max_x = max(bbox[0] + bbox[2] for bbox in bboxes)
            max_y = max(bbox[1] + bbox[3] for bbox in bboxes)
            
            defect_w = max_x - min_x
            defect_h = max_y - min_y
            long_side = max(defect_w, defect_h)
            
            if long_side > max_defect_long_side:
                stats['skipped_defect_too_large'] += 1
                continue
        
        # 如果启用了裁剪，在收集候选样本时先检查裁剪约束
        if crop_k_multiplier > 0 and crop_min_size > 0:
            h, w = img.shape[:2]
            crop_region = calculate_crop_region(bboxes, w, h, crop_k_multiplier, crop_min_size)
            if crop_region is None:
                stats['skipped_crop_failed'] += 1
                continue
        
        if defect_type not in defect_candidates:
            defect_candidates[defect_type] = []
        defect_candidates[defect_type].append((img_id, img_path, bboxes))
    
    if max_normal_train > 0 and len(normal_train_candidates) > max_normal_train:
        np.random.shuffle(normal_train_candidates)
        normal_train_candidates = normal_train_candidates[:max_normal_train]
    if max_normal_test > 0 and len(normal_test_candidates) > max_normal_test:
        np.random.shuffle(normal_test_candidates)
        normal_test_candidates = normal_test_candidates[:max_normal_test]
    
    for img_id, img_path in tqdm(normal_train_candidates + normal_test_candidates, desc="Processing normal samples"):
        img = cv2.imread(str(img_path))
        if img is None:
            continue
        
        is_train = img_id in train_set
        
        # 如果启用了正常图片裁剪（尺寸检查已在收集候选时完成）
        if normal_crop_size > 0:
            cropped_img = random_crop_to_size(img, target_size=normal_crop_size)
            dst_img = (train_dir if is_train else test_dir) / img_path.name
            cv2.imwrite(str(dst_img), cropped_img)
        else:
            dst_img = (train_dir if is_train else test_dir) / img_path.name
            shutil.copy2(img_path, dst_img)
        
        stats['train_good' if is_train else 'test_good'] += 1
    
    for defect_type, candidates in tqdm(defect_candidates.items(), desc="Processing defect samples"):
        if max_defect_samples > 0 and len(candidates) > max_defect_samples:
            np.random.shuffle(candidates)
            candidates = candidates[:max_defect_samples]
        
        test_defect_dir = output_dir / 'test' / defect_type
        test_defect_dir.mkdir(parents=True, exist_ok=True)
        
        for img_id, img_path, bboxes in candidates:
            img = cv2.imread(str(img_path))
            if img is None:
                continue
            
            h, w = img.shape[:2]
            
            if crop_k_multiplier > 0 and crop_min_size > 0:
                # 裁剪检查已在收集候选样本时完成，这里直接进行裁剪
                crop_region = calculate_crop_region(bboxes, w, h, crop_k_multiplier, crop_min_size)
                crop_x, crop_y, crop_w, crop_h = crop_region
                cropped_img = img[crop_y:crop_y+crop_h, crop_x:crop_x+crop_w]
                
                adjusted_bboxes = []
                for bbox in bboxes:
                    adjusted_bboxes.append([bbox[0] - crop_x, bbox[1] - crop_y, bbox[2], bbox[3]])
                
                mask = np.zeros((crop_h, crop_w), dtype=np.uint8)
                for adj_bbox in adjusted_bboxes:
                    mask = cv2.bitwise_or(mask, bbox_to_mask(adj_bbox, crop_w, crop_h))
                
                dst_img = test_defect_dir / img_path.name
                cv2.imwrite(str(dst_img), cropped_img)
                
                mask_file = gt_dir / defect_type / (img_path.stem + '_mask.png')
                mask_file.parent.mkdir(parents=True, exist_ok=True)
                cv2.imwrite(str(mask_file), mask)
            else:
                dst_img = test_defect_dir / img_path.name
                shutil.copy2(img_path, dst_img)
                
                mask = np.zeros((h, w), dtype=np.uint8)
                for bbox in bboxes:
                    mask = cv2.bitwise_or(mask, bbox_to_mask(bbox, w, h))
                
                mask_file = gt_dir / defect_type / (img_path.stem + '_mask.png')
                mask_file.parent.mkdir(parents=True, exist_ok=True)
                cv2.imwrite(str(mask_file), mask)
            
            stats['defect_types'][defect_type] = stats['defect_types'].get(defect_type, 0) + 1
            defect_files.setdefault(defect_type, []).append((dst_img, mask_file))
    
    if min_samples > 0:
        for dt in [dt for dt, c in stats['defect_types'].items() if c < min_samples]:
            stats['filtered_types'][dt] = stats['defect_types'].pop(dt)
            if dt in defect_files:
                for f1, f2 in defect_files[dt]:
                    f1.exists() and f1.unlink()
                    f2.exists() and f2.unlink()
                for d in [output_dir / 'test' / dt, gt_dir / dt]:
                    d.exists() and not any(d.iterdir()) and d.rmdir()
    
    print("\n" + "=" * 60)
    print("Conversion completed! Statistics:")
    print("=" * 60)
    print(f"Train normal samples: {stats['train_good']}")
    print(f"Test normal samples: {stats['test_good']}")
    print(f"Skipped multi-defect images: {stats['skipped_multi_defect']}")
    if max_defect_long_side > 0:
        print(f"Skipped images due to defect long side > {max_defect_long_side}: {stats['skipped_defect_too_large']}")
    if crop_k_multiplier > 0 and crop_min_size > 0:
        print(f"Skipped images due to crop constraints: {stats['skipped_crop_failed']}")
    if normal_crop_size > 0:
        print(f"Skipped normal images due to size < {normal_crop_size}: {stats['skipped_normal_too_small']}")
    if stats['filtered_types']:
        print(f"Filtered types (insufficient samples): {len(stats['filtered_types'])}")
    print(f"\nDefect type statistics:")
    for dt, c in sorted(stats['defect_types'].items()):
        print(f"  {dt}: {c} images")
    if stats['filtered_types']:
        print(f"\nFiltered types (samples < {min_samples}):")
        for dt, c in sorted(stats['filtered_types'].items()):
            print(f"  {dt}: {c} images")
    print(f"\nOutput directory: {output_dir}")
    print("=" * 60)
    
    write_readme(output_dir, coco_dirs, stats, target_categories, train_ratio,
                background_class, min_samples, max_normal_train, max_normal_test,
                max_defect_samples, name_mapping_file, crop_k_multiplier, crop_min_size, 
                name_mapping, normal_crop_size, max_defect_long_side)
    print(f"\nREADME.md已生成: {output_dir / 'README.md'}")
    
    # 生成 name-anomaly.txt 文件
    if category_prefix and stats['defect_types']:
        anomaly_file = original_output_dir / 'name-anomaly.txt'
        with open(anomaly_file, 'w', encoding='utf-8') as f:
            for defect_type in sorted(stats['defect_types'].keys()):
                f.write(f"{category_prefix}+{defect_type}\n")
        print(f"name-anomaly.txt已生成: {anomaly_file}")


def main():
    parser = argparse.ArgumentParser(description='Convert COCO format to MVTec dataset format')
    parser.add_argument('--dirs', nargs='+', required=True, help='COCO dataset directory list (each directory should contain _annotations.coco.json)')
    parser.add_argument('--output', required=True, help='Output directory')
    parser.add_argument('--exclude-categories', nargs='+', help='Category names to exclude (all categories will be converted if not specified)')
    parser.add_argument('--include-categories', nargs='+', help='Category names to include (all categories will be converted if not specified). If specified, only these categories will be included.')
    parser.add_argument('--train-ratio', type=float, default=0.8, help='Train set ratio (default: 0.8)')
    parser.add_argument('--background-class', default='good', help='Background/normal class name (default: good)')
    parser.add_argument('--min-samples', type=int, default=0, help='Minimum samples per defect type, types with fewer samples will be filtered (default: 0, no filtering)')
    parser.add_argument('--max-normal-train', type=int, default=0, help='Maximum normal samples in train set (default: 0, no limit)')
    parser.add_argument('--max-normal-test', type=int, default=0, help='Maximum normal samples in test set (default: 0, no limit)')
    parser.add_argument('--max-defect-samples', type=int, default=0, help='Maximum samples per defect type (default: 0, no limit)')
    parser.add_argument('--name-mapping', type=str, help='JSON file with category name mappings (format: {"original_name": "english_name"})')
    parser.add_argument('--crop-k-multiplier', type=float, default=0, help='Crop region multiplier for short side (K). If > 0, enables cropping. Crop size = max(short_side*K, long_side, min_size)')
    parser.add_argument('--crop-min-size', type=int, default=0, help='Minimum crop size in pixels (N). If > 0, enables cropping. Crop size = max(short_side*K, long_side, min_size)')
    parser.add_argument('--normal-crop-size', type=int, default=0, help='Crop size for normal images. If > 0, normal images will be randomly cropped to this size (default: 0, no cropping)')
    parser.add_argument('--max-defect-long-side', type=int, default=0, help='Maximum defect long side in pixels. If > 0, defects with long side exceeding this value will be filtered (default: 0, no filtering)')
    parser.add_argument('--category-prefix', type=str, default=None, help='Category prefix for output directory structure and name-anomaly.txt (e.g., "gear"). If specified, output will be in output_dir/category_prefix/ and name-anomaly.txt will be generated')
    
    args = parser.parse_args()
    coco_to_mvtec(args.dirs, Path(args.output), args.exclude_categories, args.train_ratio, 
                  args.background_class, args.min_samples, args.max_normal_train, 
                  args.max_normal_test, args.max_defect_samples, args.name_mapping,
                  args.crop_k_multiplier, args.crop_min_size, args.normal_crop_size, args.max_defect_long_side,
                  args.include_categories, args.category_prefix)


if __name__ == '__main__':
    main()
