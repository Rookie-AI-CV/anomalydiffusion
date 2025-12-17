#!/usr/bin/env python3
"""
图片resize脚本 - 保持宽高比并resize到指定尺寸
支持单张图片或批量处理
"""

import argparse
import os
from pathlib import Path
from PIL import Image


def resize_image(input_path, output_path, target_size, keep_aspect=True, padding_color=(255, 255, 255)):
    """
    将图片resize到指定尺寸，保持宽高比
    
    Args:
        input_path: 输入图片路径
        output_path: 输出图片路径
        target_size: 目标尺寸 (width, height)
        keep_aspect: 是否保持宽高比（默认True）
        padding_color: 填充颜色（RGB格式，默认白色）
    """
    # 打开图片
    img = Image.open(input_path)
    original_size = img.size
    target_width, target_height = target_size
    
    if keep_aspect:
        # 计算缩放比例，保持宽高比
        scale = min(target_width / original_size[0], target_height / original_size[1])
        new_width = int(original_size[0] * scale)
        new_height = int(original_size[1] * scale)
        
        # 先resize图片
        img_resized = img.resize((new_width, new_height), Image.Resampling.LANCZOS)
        
        # 创建目标尺寸的空白图片
        img_final = Image.new(img.mode, (target_width, target_height), padding_color)
        
        # 计算居中位置
        paste_x = (target_width - new_width) // 2
        paste_y = (target_height - new_height) // 2
        
        # 将resize后的图片粘贴到中心
        img_final.paste(img_resized, (paste_x, paste_y))
    else:
        # 不保持宽高比，直接拉伸
        img_final = img.resize((target_width, target_height), Image.Resampling.LANCZOS)
    
    # 保存图片
    img_final.save(output_path)
    print(f"已处理: {input_path} -> {output_path} ({original_size} -> {target_size})")


def main():
    parser = argparse.ArgumentParser(
        description='将图片resize到指定尺寸（保持宽高比）',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  # 单张图片
  python resize_image.py input.jpg output.jpg --size 512 512
  
  # 批量处理目录
  python resize_image.py input_dir/ output_dir/ --size 512 512 --batch
  
  # 不保持宽高比（拉伸）
  python resize_image.py input.jpg output.jpg --size 512 512 --no-keep-aspect
        """
    )
    
    parser.add_argument('input', help='输入图片路径或目录')
    parser.add_argument('output', help='输出图片路径或目录')
    parser.add_argument('--size', type=int, nargs=2, metavar=('WIDTH', 'HEIGHT'),
                       required=True, help='目标尺寸（宽 高）')
    parser.add_argument('--batch', action='store_true',
                       help='批量处理模式（输入和输出都是目录）')
    parser.add_argument('--no-keep-aspect', action='store_true',
                       help='不保持宽高比（直接拉伸到目标尺寸）')
    parser.add_argument('--padding-color', type=int, nargs=3, metavar=('R', 'G', 'B'),
                       default=[255, 255, 255], help='填充颜色 RGB值（默认：255 255 255 白色）')
    parser.add_argument('--extensions', nargs='+', 
                       default=['.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.webp'],
                       help='支持的图片格式（默认：jpg jpeg png bmp tiff webp）')
    
    args = parser.parse_args()
    
    target_size = tuple(args.size)
    keep_aspect = not args.no_keep_aspect
    padding_color = tuple(args.padding_color)
    
    # 支持的图片格式
    extensions = [ext.lower() if ext.startswith('.') else f'.{ext.lower()}' 
                  for ext in args.extensions]
    
    if args.batch:
        # 批量处理模式
        input_dir = Path(args.input)
        output_dir = Path(args.output)
        
        if not input_dir.exists():
            print(f"错误: 输入目录不存在: {input_dir}")
            return
        
        # 创建输出目录
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # 查找所有图片文件
        image_files = []
        for ext in extensions:
            image_files.extend(input_dir.glob(f'*{ext}'))
            image_files.extend(input_dir.glob(f'*{ext.upper()}'))
        
        if not image_files:
            print(f"错误: 在 {input_dir} 中未找到支持的图片文件")
            return
        
        print(f"找到 {len(image_files)} 张图片，开始处理...")
        
        for img_path in image_files:
            output_path = output_dir / img_path.name
            try:
                resize_image(img_path, output_path, target_size, keep_aspect, padding_color)
            except Exception as e:
                print(f"处理失败 {img_path}: {e}")
        
        print(f"\n处理完成！共处理 {len(image_files)} 张图片")
    else:
        # 单张图片模式
        input_path = Path(args.input)
        output_path = Path(args.output)
        
        if not input_path.exists():
            print(f"错误: 输入文件不存在: {input_path}")
            return
        
        # 创建输出目录（如果不存在）
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        try:
            resize_image(input_path, output_path, target_size, keep_aspect, padding_color)
            print("处理完成！")
        except Exception as e:
            print(f"处理失败: {e}")


if __name__ == '__main__':
    main()

