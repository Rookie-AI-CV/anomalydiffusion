import os
import sys
import json
import base64
import io
import torch
import numpy as np
from PIL import Image
from flask import Flask, request, jsonify, render_template, send_from_directory
from werkzeug.utils import secure_filename
from omegaconf import OmegaConf
from ldm.util import instantiate_from_config
from ldm.models.diffusion.ddim import DDIMSampler
from torchvision.utils import save_image
import tempfile
import shutil

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['MAX_CONTENT_LENGTH'] = 100 * 1024 * 1024  # 100MB max file size
app.config['SECRET_KEY'] = 'your-secret-key-here'

# 创建必要的目录
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs('static/results', exist_ok=True)
os.makedirs('static/temp', exist_ok=True)

# 全局变量存储模型
model_cache = {}
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def load_model_from_config(config, ckpt, verbose=False):
    """加载模型"""
    print(f"Loading model from {ckpt}")
    pl_sd = torch.load(ckpt, map_location="cpu")
    sd = pl_sd["state_dict"]
    config.model.params.ckpt_path = ckpt
    model = instantiate_from_config(config.model)
    m, u = model.load_state_dict(sd, strict=False)
    if len(m) > 0 and verbose:
        print("missing keys:")
        print(m)
    if len(u) > 0 and verbose:
        print("unexpected keys:")
        print(u)
    return model

def make_batch_from_images(image_array, mask_array, device):
    """从numpy数组创建batch"""
    # 处理图像
    if isinstance(image_array, Image.Image):
        image_array = np.array(image_array.convert("RGB"))
    
    image = image_array.astype(np.float32) / 255.0
    if len(image.shape) == 3:
        # HWC -> CHW -> BCHW
        image = image[None].transpose(0, 3, 1, 2)
    image = torch.from_numpy(image).to(device) * 2 - 1
    
    # 处理掩码
    if isinstance(mask_array, Image.Image):
        mask_array = np.array(mask_array.convert("L"))
    
    mask = mask_array.astype(np.float32) / 255.0
    if len(mask.shape) == 2:
        # HW -> 1HW
        mask = mask[None, None, :, :]
    elif len(mask.shape) == 3:
        # HWC -> CHW -> 1CHW
        mask = mask[None].transpose(0, 3, 1, 2)
        mask = mask[:, 0:1, :, :]  # 只取第一个通道
    else:
        mask = mask[None]
    
    mask[mask < 0.5] = 0
    mask[mask >= 0.5] = 1
    mask = torch.from_numpy(mask).to(device)
    
    batch = {"image": image, "mask": mask}
    return batch

def create_mask_from_selection(image_shape, selection_coords):
    """从框选坐标创建掩码"""
    h, w = image_shape[:2]
    mask = np.zeros((h, w), dtype=np.uint8)
    
    if selection_coords:
        x1, y1, x2, y2 = selection_coords
        # 确保坐标在图像范围内
        x1, y1 = max(0, int(x1)), max(0, int(y1))
        x2, y2 = min(w, int(x2)), min(h, int(y2))
        mask[y1:y2, x1:x2] = 255
    
    return mask

@app.route('/')
def index():
    """主页面"""
    return render_template('index.html')

def load_model_helper(model_path, config_path, spatial_encoder_path, embeddings_path):
    """加载模型的辅助函数"""
    # 检查模型文件是否存在
    if not os.path.exists(model_path):
        raise FileNotFoundError(f'模型文件不存在: {model_path}')
    
    # 加载配置
    if not os.path.exists(config_path):
        raise FileNotFoundError(f'配置文件不存在: {config_path}')
    
    config = OmegaConf.load(config_path)
    
    # 加载模型
    model = load_model_from_config(config, model_path)
    model = model.to(device)
    
    # 准备spatial encoder
    model.prepare_spatial_encoder(optimze_together=True)
    
    # 加载spatial encoder
    if spatial_encoder_path and os.path.exists(spatial_encoder_path):
        ckpt = torch.load(spatial_encoder_path, map_location="cpu")
        model.embedding_manager.spatial_encoder_model.load_state_dict(ckpt)
        print(f"Loaded spatial encoder from {spatial_encoder_path}")
    elif spatial_encoder_path:
        print(f"Warning: spatial encoder path provided but file not found: {spatial_encoder_path}")
    
    # 加载embeddings
    if embeddings_path and os.path.exists(embeddings_path):
        model.embedding_manager.load(embeddings_path)
        print(f"Loaded embeddings from {embeddings_path}")
    elif embeddings_path:
        print(f"Warning: embeddings path provided but file not found: {embeddings_path}")
    
    return model, config

@app.route('/api/upload_model', methods=['POST'])
def upload_model():
    """上传模型文件"""
    try:
        if 'model_file' not in request.files:
            return jsonify({'error': '没有上传模型文件'}), 400
        
        file = request.files['model_file']
        if file.filename == '':
            return jsonify({'error': '文件名为空'}), 400
        
        # 保存文件
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        
        # 获取其他参数
        config_path = request.form.get('config_path', 'configs/latent-diffusion/txt2img-1p4B-finetune-encoder+embedding.yaml')
        spatial_encoder_path = request.form.get('spatial_encoder_path', '')
        embeddings_path = request.form.get('embeddings_path', '')
        
        # 加载模型
        model, config = load_model_helper(filepath, config_path, spatial_encoder_path, embeddings_path)
        
        # 缓存模型
        model_id = filename
        model_cache[model_id] = {
            'model': model,
            'config': config,
            'spatial_encoder_path': spatial_encoder_path,
            'embeddings_path': embeddings_path,
            'model_path': filepath
        }
        
        return jsonify({
            'success': True,
            'model_id': model_id,
            'message': '模型加载成功'
        })
    
    except Exception as e:
        return jsonify({'error': f'加载模型失败: {str(e)}'}), 500

@app.route('/api/load_model_from_path', methods=['POST'])
def load_model_from_path():
    """从服务器路径加载模型"""
    try:
        data = request.get_json()
        
        model_path = data.get('model_path', '').strip()
        config_path = data.get('config_path', 'configs/latent-diffusion/txt2img-1p4B-finetune-encoder+embedding.yaml').strip()
        spatial_encoder_path = data.get('spatial_encoder_path', '').strip()
        embeddings_path = data.get('embeddings_path', '').strip()
        
        if not model_path:
            return jsonify({'error': '请提供模型文件路径'}), 400
        
        # 加载模型
        model, config = load_model_helper(model_path, config_path, spatial_encoder_path, embeddings_path)
        
        # 使用路径作为model_id（使用绝对路径的hash避免冲突）
        import hashlib
        model_id = hashlib.md5(model_path.encode()).hexdigest()[:16]
        
        # 如果已存在相同路径的模型，先清除
        for key, value in list(model_cache.items()):
            if value.get('model_path') == model_path:
                del model_cache[key]
        
        # 缓存模型
        model_cache[model_id] = {
            'model': model,
            'config': config,
            'spatial_encoder_path': spatial_encoder_path,
            'embeddings_path': embeddings_path,
            'model_path': model_path
        }
        
        return jsonify({
            'success': True,
            'model_id': model_id,
            'message': f'模型加载成功: {model_path}'
        })
    
    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({'error': f'加载模型失败: {str(e)}'}), 500

@app.route('/api/parse_anomaly_file', methods=['POST'])
def parse_anomaly_file():
    """解析name-anomaly.txt文件"""
    try:
        if 'anomaly_file' not in request.files:
            return jsonify({'error': '没有上传anomaly文件'}), 400
        
        file = request.files['anomaly_file']
        if file.filename == '':
            return jsonify({'error': '文件名为空'}), 400
        
        # 读取文件内容
        content = file.read().decode('utf-8')
        lines = content.strip().split('\n')
        
        # 解析缺陷类型
        anomalies = []
        for line in lines:
            line = line.strip()
            if line and '+' in line:
                parts = line.split('+')
                if len(parts) >= 2:
                    sample_name = parts[0]
                    anomaly_name = parts[1]
                    anomalies.append({
                        'sample_name': sample_name,
                        'anomaly_name': anomaly_name,
                        'display': f"{sample_name} - {anomaly_name}"
                    })
        
        return jsonify({
            'success': True,
            'anomalies': anomalies
        })
    
    except Exception as e:
        return jsonify({'error': f'解析文件失败: {str(e)}'}), 500

@app.route('/api/generate_anomaly', methods=['POST'])
def generate_anomaly():
    """生成异常图像"""
    try:
        data = request.get_json()
        
        # 获取参数
        model_id = data.get('model_id')
        image_data = data.get('image')  # base64编码的图片
        selection_coords = data.get('selection_coords')  # [x1, y1, x2, y2]
        anomaly_name = data.get('anomaly_name')
        adaptive_mask = data.get('adaptive_mask', False)
        
        if not model_id or model_id not in model_cache:
            return jsonify({'error': '模型未加载，请先上传模型'}), 400
        
        if not image_data:
            return jsonify({'error': '没有提供图片'}), 400
        
        if not selection_coords:
            return jsonify({'error': '没有选择区域'}), 400
        
        # 解码图片
        image_data = image_data.split(',')[1] if ',' in image_data else image_data
        image_bytes = base64.b64decode(image_data)
        image = Image.open(io.BytesIO(image_bytes)).convert('RGB')
        
        # 保存原始尺寸用于后续显示
        original_size = image.size
        
        # 创建掩码（在原始尺寸上）
        mask_array = create_mask_from_selection(np.array(image), selection_coords)
        mask_image = Image.fromarray(mask_array).convert('L')
        
        # Resize到模型期望的尺寸 (256x256)
        target_size = (256, 256)
        # 兼容不同PIL版本
        try:
            resize_method = Image.Resampling.LANCZOS
        except AttributeError:
            resize_method = Image.LANCZOS
        image_resized = image.resize(target_size, resize_method)
        mask_resized = mask_image.resize(target_size, resize_method)
        
        # 转换为numpy数组
        image_array = np.array(image_resized)
        mask_array = np.array(mask_resized)
        
        # 创建batch
        batch = make_batch_from_images(image_array, mask_array, device)
        
        # 获取模型
        model_info = model_cache[model_id]
        model = model_info['model']
        
        # 生成异常图像
        with torch.no_grad():
            with model.ema_scope():
                images = model.log_images(
                    batch,
                    sample=False,
                    inpaint=True,
                    unconditional_only=True,
                    adaptive_mask=adaptive_mask
                )
                
                generated_img = images['samples_inpainting'].cpu()
                recon_img = images.get('reconstruction', None)
                
                # 处理生成的图像
                generated_img = (generated_img[0] + 1) / 2
                generated_img = torch.clamp(generated_img, 0, 1)
                
                # 转换为PIL图像
                generated_array = (generated_img.permute(1, 2, 0).numpy() * 255).astype(np.uint8)
                generated_pil = Image.fromarray(generated_array)
                
                # Resize回原始尺寸以便显示
                try:
                    resize_method = Image.Resampling.LANCZOS
                except AttributeError:
                    resize_method = Image.LANCZOS
                generated_pil = generated_pil.resize(original_size, resize_method)
                
                # 保存到临时文件
                result_id = f"result_{hash(str(selection_coords)) % 100000}"
                result_path = f'static/results/{result_id}.jpg'
                generated_pil.save(result_path)
                
                # 转换为base64返回
                buffered = io.BytesIO()
                generated_pil.save(buffered, format="JPEG")
                img_base64 = base64.b64encode(buffered.getvalue()).decode()
                
                # 保存掩码
                mask_path = f'static/results/{result_id}_mask.jpg'
                mask_image.save(mask_path)
                
                return jsonify({
                    'success': True,
                    'result_image': f'data:image/jpeg;base64,{img_base64}',
                    'result_path': f'/static/results/{result_id}.jpg',
                    'mask_path': f'/static/results/{result_id}_mask.jpg'
                })
    
    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({'error': f'生成失败: {str(e)}'}), 500

@app.route('/static/<path:filename>')
def static_files(filename):
    """提供静态文件"""
    return send_from_directory('static', filename)

if __name__ == '__main__':
    print(f"使用设备: {device}")
    print("Flask应用启动中...")
    app.run(host='0.0.0.0', port=5000, debug=True)
