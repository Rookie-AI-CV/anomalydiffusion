#!/bin/bash

# Flask异常生成工具启动脚本

echo "=========================================="
echo "启动Flask异常生成工具"
echo "=========================================="

# 检查Python环境
if ! command -v python &> /dev/null; then
    echo "错误: 未找到Python，请先安装Python"
    exit 1
fi

# 检查Flask是否安装
python -c "import flask" 2>/dev/null
if [ $? -ne 0 ]; then
    echo "警告: Flask未安装，正在安装..."
    pip install flask>=2.0.0
fi

# 创建必要的目录
mkdir -p uploads static/results static/temp templates

# 检查模板文件是否存在
if [ ! -f "templates/index.html" ]; then
    echo "错误: 未找到模板文件 templates/index.html"
    exit 1
fi

# 设置GPU（如果可用）
if command -v nvidia-smi &> /dev/null; then
    echo "检测到GPU，将使用CUDA"
    export CUDA_VISIBLE_DEVICES=0
else
    echo "未检测到GPU，将使用CPU（速度较慢）"
fi

# 启动Flask应用
echo ""
echo "Flask服务启动中..."
echo "访问地址: http://localhost:5000"
echo "按 Ctrl+C 停止服务"
echo ""

python flask_app.py

