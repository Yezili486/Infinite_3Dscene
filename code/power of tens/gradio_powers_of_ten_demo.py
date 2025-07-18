#!/usr/bin/env python3
"""
Generative Powers of Ten - Interactive Gradio Demo

基于论文 "Generative Powers of Ten" 的交互式演示界面
支持多尺度缩放栈生成、实时图像浏览和动画播放

运行方式：
    python gradio_powers_of_ten_demo.py

依赖包：
    pip install gradio torch torchvision opencv-python pillow numpy matplotlib
"""

import gradio as gr
import torch
import numpy as np
import cv2
import warnings
import time
import os
from PIL import Image
import matplotlib.pyplot as plt
from typing import List, Tuple, Optional
import threading
import math

warnings.filterwarnings("ignore")

# 设备配置
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

# 全局变量存储当前的缩放栈
current_zoom_stack = None
generation_status = {"status": "idle", "progress": 0, "message": ""}

# ==================== 核心数据结构 ====================

class ZoomStack:
    """缩放栈数据结构"""
    
    def __init__(self, zoom_factors, H=512, W=512, device="cpu"):
        self.zoom_factors = zoom_factors
        self.N = len(zoom_factors)
        self.H = H
        self.W = W
        self.device = device
        
        # 验证缩放因子
        for i, p in enumerate(zoom_factors):
            if i == 0:
                assert p == 1, "第一个缩放因子必须是1"
            else:
                assert p == 2 * zoom_factors[i-1], f"缩放因子必须是2的幂序列，第{i}个因子{p}不符合要求"
        
        # 初始化层
        self.layers = self._initialize_layers()
        
    def _initialize_layers(self):
        """初始化所有层"""
        layers = []
        for i, p in enumerate(self.zoom_factors):
            layer = torch.randn(self.H, self.W, 3, device=self.device, dtype=torch.float32)
            layer = layer * 0.1  # 减小初始噪声幅度
            layers.append(layer)
        return layers
    
    def get_layer(self, i):
        """获取第 i 层"""
        assert 0 <= i < self.N, f"层索引 {i} 超出范围 [0, {self.N-1}]"
        return self.layers[i]
    
    def set_layer(self, i, layer):
        """设置第 i 层"""
        assert 0 <= i < self.N, f"层索引 {i} 超出范围 [0, {self.N-1}]"
        assert layer.shape == (self.H, self.W, 3), f"层形状必须是 ({self.H}, {self.W}, 3)"
        self.layers[i] = layer.to(self.device)
    
    def get_zoom_factor(self, i):
        """获取第 i 层的缩放因子"""
        return self.zoom_factors[i]
    
    def get_all_layers(self):
        """获取所有层的副本"""
        return [layer.clone() for layer in self.layers]
    
    def to_image_format(self, layer_idx):
        """将层转换为图像格式 [0, 1]"""
        layer = self.get_layer(layer_idx)
        return torch.clamp((layer + 1.0) / 2.0, 0.0, 1.0)

def create_zoom_stack(zoom_factors, H=512, W=512, device="cpu"):
    """创建缩放栈的便捷函数"""
    return ZoomStack(zoom_factors, H, W, device)

def generate_zoom_factors(N):
    """生成 N 个缩放因子：[1, 2, 4, ..., 2^{N-1}]"""
    return [2**i for i in range(N)]

# ==================== 渲染函数 ====================

def Pi_image(zoom_stack, i):
    """从缩放栈渲染图像（算法 1）"""
    rendered = zoom_stack.get_layer(i).clone()
    H, W = rendered.shape[:2]
    p_i = zoom_stack.get_zoom_factor(i)
    
    # 对于所有更高层，下采样并融合
    for j in range(i + 1, zoom_stack.N):
        layer_j = zoom_stack.get_layer(j)
        p_j = zoom_stack.get_zoom_factor(j)
        
        # 中心裁剪
        crop_h = max(1, H // p_j)
        crop_w = max(1, W // p_j)
        start_h = (H - crop_h) // 2
        start_w = (W - crop_w) // 2
        end_h = start_h + crop_h
        end_w = start_w + crop_w
        
        cropped_j = layer_j[start_h:end_h, start_w:end_w, :]
        
        # 上采样
        if crop_h != H or crop_w != W:
            cropped_j_chw = cropped_j.permute(2, 0, 1).unsqueeze(0)
            upsampled_j = torch.nn.functional.interpolate(
                cropped_j_chw, 
                size=(H, W), 
                mode='bilinear', 
                align_corners=False
            )
            upsampled_j = upsampled_j.squeeze(0).permute(1, 2, 0)
        else:
            upsampled_j = cropped_j
        
        # 融合掩码（中心区域）
        mask = torch.zeros((H, W, 1), device=zoom_stack.device, dtype=torch.float32)
        mask_h_start = (H - crop_h * p_j) // 2
        mask_w_start = (W - crop_w * p_j) // 2
        mask_h_end = min(H, mask_h_start + crop_h * p_j)
        mask_w_end = min(W, mask_w_start + crop_w * p_j)
        
        if mask_h_end > mask_h_start and mask_w_end > mask_w_start:
            mask[mask_h_start:mask_h_end, mask_w_start:mask_w_end, :] = 1.0
        
        rendered = rendered * (1 - mask) + upsampled_j * mask
    
    return rendered

def Pi_noise(zoom_stack, i):
    """从缩放栈渲染噪声（算法 1）"""
    H, W = zoom_stack.H, zoom_stack.W
    rendered_noise = torch.zeros((H, W, 3), device=zoom_stack.device, dtype=torch.float32)
    p_i = zoom_stack.get_zoom_factor(i)
    total_variance = torch.zeros((H, W, 1), device=zoom_stack.device, dtype=torch.float32)
    
    for j in range(i, zoom_stack.N):
        p_j = zoom_stack.get_zoom_factor(j)
        noise_scale = p_j / p_i
        
        if j == i:
            layer_noise = torch.randn((H, W, 3), device=zoom_stack.device, dtype=torch.float32)
            layer_noise *= noise_scale
            mask = torch.ones((H, W, 1), device=zoom_stack.device, dtype=torch.float32)
        else:
            crop_h = max(1, H // p_j)
            crop_w = max(1, W // p_j)
            
            cropped_noise = torch.randn((crop_h, crop_w, 3), device=zoom_stack.device, dtype=torch.float32)
            cropped_noise *= noise_scale
            
            cropped_noise_chw = cropped_noise.permute(2, 0, 1).unsqueeze(0)
            upsampled_noise = torch.nn.functional.interpolate(
                cropped_noise_chw,
                size=(H, W),
                mode='bilinear',
                align_corners=False
            )
            layer_noise = upsampled_noise.squeeze(0).permute(1, 2, 0)
            
            mask = torch.zeros((H, W, 1), device=zoom_stack.device, dtype=torch.float32)
            mask_h_start = (H - crop_h * p_j) // 2
            mask_w_start = (W - crop_w * p_j) // 2
            mask_h_end = min(H, mask_h_start + crop_h * p_j)
            mask_w_end = min(W, mask_w_start + crop_w * p_j)
            
            if mask_h_end > mask_h_start and mask_w_end > mask_w_start:
                mask[mask_h_start:mask_h_end, mask_w_start:mask_w_end, :] = 1.0
            
            layer_noise = layer_noise * mask
        
        rendered_noise += layer_noise
        total_variance += mask * (noise_scale ** 2)
    
    std_dev = torch.sqrt(total_variance + 1e-8)
    rendered_noise = rendered_noise / std_dev
    
    return rendered_noise

# ==================== 简化的生成函数 ====================

def create_artistic_pattern(prompt, H, W, scale_index, total_scales):
    """基于提示创建艺术图案（简化版本，不需要AI模型）"""
    pattern = torch.zeros((H, W, 3), dtype=torch.float32)
    
    # 根据提示关键词创建不同的图案
    prompt_lower = prompt.lower()
    
    # 创建基础坐标网格
    y_coords, x_coords = torch.meshgrid(
        torch.linspace(-1, 1, H), 
        torch.linspace(-1, 1, W), 
        indexing='ij'
    )
    r = torch.sqrt(x_coords**2 + y_coords**2)
    theta = torch.atan2(y_coords, x_coords)
    
    # 根据不同关键词生成不同图案
    if any(word in prompt_lower for word in ['galaxy', 'spiral', 'cosmic', 'space']):
        # 螺旋星系图案
        spiral = torch.sin(3 * theta + 5 * r) * torch.exp(-r * 2)
        pattern[:, :, 0] = 0.3 + 0.4 * spiral  # 红色
        pattern[:, :, 1] = 0.1 + 0.2 * spiral  # 绿色
        pattern[:, :, 2] = 0.6 + 0.3 * spiral  # 蓝色
        
    elif any(word in prompt_lower for word in ['star', 'system', 'planet']):
        # 行星系统图案
        for i in range(3):
            orbit_radius = 0.2 + i * 0.3
            orbit = torch.exp(-((r - orbit_radius) ** 2) / 0.01)
            pattern[:, :, i] = 0.2 + 0.5 * orbit
            
    elif any(word in prompt_lower for word in ['surface', 'mountain', 'terrain', 'landscape']):
        # 地形图案
        terrain = torch.sin(x_coords * 8) * torch.sin(y_coords * 6) * torch.exp(-r)
        pattern[:, :, 0] = 0.4 + 0.3 * terrain  # 土地色
        pattern[:, :, 1] = 0.3 + 0.4 * terrain  # 绿色
        pattern[:, :, 2] = 0.2 + 0.2 * terrain  # 蓝色
        
    elif any(word in prompt_lower for word in ['tree', 'forest', 'branch', 'bark']):
        # 树木/森林图案
        tree_pattern = torch.sin(y_coords * 10) * torch.cos(x_coords * 8)
        pattern[:, :, 0] = 0.2 + 0.3 * torch.abs(tree_pattern)  # 棕色
        pattern[:, :, 1] = 0.4 + 0.4 * torch.abs(tree_pattern)  # 绿色
        pattern[:, :, 2] = 0.1 + 0.2 * torch.abs(tree_pattern)  # 蓝色
        
    elif any(word in prompt_lower for word in ['insect', 'bug', 'detail', 'micro']):
        # 微观细节图案
        detail = torch.sin(x_coords * 20) * torch.sin(y_coords * 20) * torch.exp(-r * 0.5)
        pattern[:, :, 0] = 0.5 + 0.3 * detail
        pattern[:, :, 1] = 0.3 + 0.4 * detail
        pattern[:, :, 2] = 0.2 + 0.3 * detail
        
    elif any(word in prompt_lower for word in ['ocean', 'water', 'wave', 'sea']):
        # 海洋图案
        waves = torch.sin(x_coords * 6 + theta * 3) * torch.cos(y_coords * 4)
        pattern[:, :, 0] = 0.1 + 0.2 * torch.abs(waves)  # 红色（少）
        pattern[:, :, 1] = 0.3 + 0.3 * torch.abs(waves)  # 绿色
        pattern[:, :, 2] = 0.6 + 0.3 * torch.abs(waves)  # 蓝色（海洋色）
        
    elif any(word in prompt_lower for word in ['coral', 'reef', 'underwater']):
        # 珊瑚礁图案
        coral = torch.sin(r * 15) * torch.cos(theta * 8) * torch.exp(-r)
        pattern[:, :, 0] = 0.6 + 0.3 * coral  # 珊瑚色
        pattern[:, :, 1] = 0.2 + 0.4 * coral
        pattern[:, :, 2] = 0.4 + 0.3 * coral
        
    elif any(word in prompt_lower for word in ['fish', 'tropical', 'close']):
        # 鱼类图案
        fish_body = torch.exp(-((x_coords - 0.2)**2 + y_coords**2) / 0.1)
        pattern[:, :, 0] = 0.7 + 0.2 * fish_body  # 橙色
        pattern[:, :, 1] = 0.5 + 0.3 * fish_body  # 黄色
        pattern[:, :, 2] = 0.2 + 0.2 * fish_body
        
    else:
        # 默认抽象图案
        default_pattern = torch.sin(r * 10 + theta * scale_index)
        pattern[:, :, 0] = 0.5 + 0.3 * default_pattern
        pattern[:, :, 1] = 0.4 + 0.3 * default_pattern
        pattern[:, :, 2] = 0.3 + 0.4 * default_pattern
    
    # 添加噪声以增加真实感
    noise = torch.randn_like(pattern) * 0.05
    pattern += noise
    
    # 根据缩放级别调整细节
    detail_factor = 1.0 + scale_index * 0.5
    pattern *= detail_factor
    
    # 限制到[-1, 1]范围
    pattern = torch.clamp(pattern * 2 - 1, -1, 1)
    
    return pattern

def joint_multi_scale_sampling_demo(prompts, zoom_factors, T=15, H=512, W=512):
    """演示版本的联合多尺度采样（无需AI模型）"""
    global generation_status
    
    generation_status["status"] = "generating"
    generation_status["progress"] = 0
    generation_status["message"] = "初始化缩放栈..."
    
    N = len(prompts)
    zoom_stack = create_zoom_stack(zoom_factors, H, W, device)
    
    # 生成初始图案
    for i, prompt in enumerate(prompts):
        generation_status["progress"] = int((i / N) * 30)
        generation_status["message"] = f"生成第 {i+1}/{N} 层图案..."
        
        # 创建基于提示的艺术图案
        initial_pattern = create_artistic_pattern(prompt, H, W, i, N)
        zoom_stack.set_layer(i, initial_pattern)
        
        time.sleep(0.1)  # 模拟处理时间
    
    # 简化的扩散循环（多尺度优化）
    for t in range(T, 0, -1):
        progress = (T - t + 1) / T
        generation_status["progress"] = int(30 + progress * 70)
        generation_status["message"] = f"多尺度优化步骤 {T-t+1}/{T}"
        
        # 渲染当前状态
        for i in range(N):
            img_rendered = Pi_image(zoom_stack, i)
            noise_rendered = Pi_noise(zoom_stack, i)
            
            # 添加渐进式优化
            noise_level = t / T * 0.2  # 减小噪声强度
            denoising_strength = progress * 0.5
            
            # 混合优化
            base_img = zoom_stack.get_layer(i)
            optimized = (base_img * (1 - denoising_strength) + 
                        img_rendered * denoising_strength * 0.7 +
                        noise_rendered * noise_level * 0.1)
            
            zoom_stack.set_layer(i, optimized)
        
        time.sleep(0.05)  # 模拟处理时间
    
    generation_status["status"] = "completed"
    generation_status["progress"] = 100
    generation_status["message"] = "生成完成!"
    
    return zoom_stack

# ==================== Gradio 界面函数 ====================

def tensor_to_pil(tensor):
    """将PyTorch张量转换为PIL图像"""
    # 转换到[0, 1]范围
    img_tensor = torch.clamp((tensor + 1.0) / 2.0, 0.0, 1.0)
    # 转换为numpy
    img_numpy = (img_tensor.cpu().numpy() * 255).astype(np.uint8)
    # 转换为PIL图像
    pil_img = Image.fromarray(img_numpy)
    return pil_img

def parse_prompts(prompts_text):
    """解析提示文本"""
    if not prompts_text.strip():
        return []
    
    # 按行分割或按逗号分割
    if '\n' in prompts_text:
        prompts = [p.strip() for p in prompts_text.split('\n') if p.strip()]
    else:
        prompts = [p.strip() for p in prompts_text.split(',') if p.strip()]
    
    return prompts

def parse_zoom_factors(zoom_factors_text):
    """解析缩放因子"""
    try:
        if not zoom_factors_text.strip():
            return []
        
        # 支持多种格式
        zoom_factors_text = zoom_factors_text.replace('[', '').replace(']', '').replace('(', '').replace(')', '')
        factors = [int(f.strip()) for f in zoom_factors_text.split(',') if f.strip()]
        
        # 验证是否是2的幂序列
        if factors and factors[0] != 1:
            raise ValueError("第一个缩放因子必须是1")
        
        for i in range(1, len(factors)):
            if factors[i] != 2 * factors[i-1]:
                raise ValueError(f"缩放因子必须是2的幂序列，第{i}个因子{factors[i]}不符合要求")
        
        return factors
    except Exception as e:
        raise ValueError(f"缩放因子格式错误: {e}")

def generate_zoom_stack_ui(prompts_text, zoom_factors_text, progress=gr.Progress()):
    """生成缩放栈（UI接口）"""
    global current_zoom_stack
    
    try:
        # 解析输入
        prompts = parse_prompts(prompts_text)
        zoom_factors = parse_zoom_factors(zoom_factors_text)
        
        if not prompts:
            return "❌ 错误：请输入至少一个提示", None, gr.update(visible=False)
        
        if not zoom_factors:
            return "❌ 错误：请输入有效的缩放因子", None, gr.update(visible=False)
        
        if len(prompts) != len(zoom_factors):
            return f"❌ 错误：提示数量({len(prompts)})与缩放因子数量({len(zoom_factors)})不匹配", None, gr.update(visible=False)
        
        # 生成缩放栈
        progress(0, desc="开始生成...")
        
        # 使用线程来监控进度
        def monitor_progress():
            while generation_status["status"] == "generating":
                progress(generation_status["progress"] / 100, desc=generation_status["message"])
                time.sleep(0.1)
        
        monitor_thread = threading.Thread(target=monitor_progress)
        monitor_thread.start()
        
        current_zoom_stack = joint_multi_scale_sampling_demo(
            prompts=prompts,
            zoom_factors=zoom_factors,
            T=15,  # 减少步数以提高交互性
            H=512, W=512
        )
        
        monitor_thread.join()
        progress(1.0, desc="生成完成!")
        
        # 返回第一层图像作为预览
        first_layer_img = Pi_image(current_zoom_stack, 0)
        preview_img = tensor_to_pil(first_layer_img)
        
        # 更新滑块范围
        slider_update = gr.update(
            minimum=0, 
            maximum=len(zoom_factors)-1, 
            value=0, 
            step=1,
            visible=True,
            label=f"缩放级别 (0-{len(zoom_factors)-1})"
        )
        
        return f"✅ 成功生成 {len(zoom_factors)} 层缩放栈！", preview_img, slider_update
        
    except Exception as e:
        return f"❌ 生成失败: {str(e)}", None, gr.update(visible=False)

def render_scale_image(scale_idx):
    """渲染指定尺度的图像"""
    global current_zoom_stack
    
    if current_zoom_stack is None:
        return None, "请先生成缩放栈"
    
    try:
        scale_idx = int(scale_idx)
        if scale_idx < 0 or scale_idx >= current_zoom_stack.N:
            return None, f"尺度索引超出范围 (0-{current_zoom_stack.N-1})"
        
        # 渲染图像
        rendered_img = Pi_image(current_zoom_stack, scale_idx)
        pil_img = tensor_to_pil(rendered_img)
        
        zoom_factor = current_zoom_stack.get_zoom_factor(scale_idx)
        info_text = f"尺度 {scale_idx}: 缩放因子 {zoom_factor}x (分辨率: {current_zoom_stack.H}x{current_zoom_stack.W})"
        
        return pil_img, info_text
        
    except Exception as e:
        return None, f"渲染失败: {str(e)}"

def create_zoom_animation():
    """创建缩放动画"""
    global current_zoom_stack
    
    if current_zoom_stack is None:
        return None, "请先生成缩放栈"
    
    try:
        print("开始创建缩放动画...")
        
        # 创建动画帧
        frames = []
        for i in range(current_zoom_stack.N):
            rendered_img = Pi_image(current_zoom_stack, i)
            pil_img = tensor_to_pil(rendered_img)
            frames.append(pil_img)
        
        # 保存为GIF
        if frames:
            output_path = "zoom_animation.gif"
            frames[0].save(
                output_path,
                save_all=True,
                append_images=frames[1:],
                duration=1500,  # 每帧1.5秒
                loop=0
            )
            
            print(f"动画已保存: {output_path}")
            return output_path, f"✅ 动画已创建，包含 {len(frames)} 帧 (每帧1.5秒)"
        else:
            return None, "❌ 无法创建动画帧"
            
    except Exception as e:
        print(f"动画创建失败: {e}")
        return None, f"❌ 动画创建失败: {str(e)}"

def get_example_inputs():
    """获取示例输入"""
    examples = {
        "cosmic": {
            "prompts": "distant galaxy with swirling spiral arms\nstar system with multiple planets\nplanet surface with mountains\nclose-up of alien vegetation",
            "zoom_factors": "1, 2, 4, 8"
        },
        "nature": {
            "prompts": "vast forest landscape\ntree with many branches\nbark texture detail\ninsect on bark surface",
            "zoom_factors": "1, 2, 4, 8"
        },
        "ocean": {
            "prompts": "ocean from space\nocean surface with waves\nunderwater coral reef\ntropical fish close-up",
            "zoom_factors": "1, 2, 4, 8"
        },
        "simple": {
            "prompts": "abstract art\ncolorful pattern\ngeometric design",
            "zoom_factors": "1, 2, 4"
        }
    }
    return examples

# ==================== Gradio 界面 ====================

def create_gradio_interface():
    """创建Gradio界面"""
    
    with gr.Blocks(
        title="Generative Powers of Ten - Interactive Demo",
        theme=gr.themes.Soft(),
        css="""
        .main-container { max-width: 1200px; margin: auto; }
        .header { text-align: center; margin-bottom: 2rem; }
        .status-text { font-weight: bold; }
        .example-btn { margin: 2px; }
        """
    ) as demo:
        
        # 标题和说明
        gr.HTML("""
        <div class="header">
            <h1>🔍 Generative Powers of Ten - Interactive Demo</h1>
            <p>基于论文 "Generative Powers of Ten" 的交互式演示</p>
            <p>输入文本提示生成多尺度缩放栈，并实时浏览不同缩放级别的图像</p>
            <p><em>演示版本：使用程序化图案生成，无需下载AI模型</em></p>
        </div>
        """)
        
        with gr.Row():
            # 左侧：输入控制面板
            with gr.Column(scale=1):
                gr.Markdown("## 📝 输入配置")
                
                prompts_input = gr.Textbox(
                    label="文本提示 (每行一个或逗号分隔)",
                    placeholder="例如：\ndistant galaxy\nstar system\nplanet surface\ninsect detail",
                    lines=6,
                    value="distant galaxy with swirling spiral arms\nstar system with multiple planets\nplanet surface with mountains\nclose-up of alien vegetation"
                )
                
                zoom_factors_input = gr.Textbox(
                    label="缩放因子 (2的幂序列，逗号分隔)",
                    placeholder="例如：1, 2, 4, 8",
                    value="1, 2, 4, 8"
                )
                
                generate_btn = gr.Button("🚀 生成缩放栈", variant="primary", size="lg")
                
                status_text = gr.Textbox(
                    label="状态",
                    value="准备就绪 - 点击生成按钮开始",
                    interactive=False,
                    elem_classes=["status-text"]
                )
                
                # 示例按钮
                gr.Markdown("### 🎯 快速示例")
                examples = get_example_inputs()
                
                with gr.Row():
                    cosmic_btn = gr.Button("🌌 宇宙缩放", size="sm", elem_classes=["example-btn"])
                    nature_btn = gr.Button("🌳 自然缩放", size="sm", elem_classes=["example-btn"])
                
                with gr.Row():
                    ocean_btn = gr.Button("🌊 海洋缩放", size="sm", elem_classes=["example-btn"])
                    simple_btn = gr.Button("🎨 简单示例", size="sm", elem_classes=["example-btn"])
            
            # 右侧：显示面板
            with gr.Column(scale=2):
                gr.Markdown("## 🖼️ 图像显示")
                
                # 主图像显示
                main_image = gr.Image(
                    label="当前尺度图像",
                    type="pil",
                    height=400,
                    show_download_button=True
                )
                
                # 缩放级别控制
                scale_slider = gr.Slider(
                    minimum=0,
                    maximum=3,
                    value=0,
                    step=1,
                    label="缩放级别",
                    visible=False,
                    interactive=True
                )
                
                scale_info = gr.Textbox(
                    label="尺度信息",
                    value="",
                    interactive=False
                )
                
                # 动画控制
                gr.Markdown("### 🎬 缩放动画")
                with gr.Row():
                    animation_btn = gr.Button("▶️ 创建缩放动画", variant="secondary")
                
                animation_output = gr.File(
                    label="动画文件 (GIF)",
                    visible=False
                )
                
                animation_status = gr.Textbox(
                    label="动画状态",
                    value="",
                    interactive=False
                )
        
        # 事件绑定
        
        # 生成缩放栈
        generate_btn.click(
            fn=generate_zoom_stack_ui,
            inputs=[prompts_input, zoom_factors_input],
            outputs=[status_text, main_image, scale_slider]
        )
        
        # 滑块变化时重新渲染
        scale_slider.change(
            fn=render_scale_image,
            inputs=[scale_slider],
            outputs=[main_image, scale_info]
        )
        
        # 创建动画
        animation_btn.click(
            fn=create_zoom_animation,
            outputs=[animation_output, animation_status]
        ).then(
            lambda: gr.update(visible=True),
            outputs=[animation_output]
        )
        
        # 示例按钮事件
        cosmic_btn.click(
            lambda: (examples["cosmic"]["prompts"], examples["cosmic"]["zoom_factors"]),
            outputs=[prompts_input, zoom_factors_input]
        )
        
        nature_btn.click(
            lambda: (examples["nature"]["prompts"], examples["nature"]["zoom_factors"]),
            outputs=[prompts_input, zoom_factors_input]
        )
        
        ocean_btn.click(
            lambda: (examples["ocean"]["prompts"], examples["ocean"]["zoom_factors"]),
            outputs=[prompts_input, zoom_factors_input]
        )
        
        simple_btn.click(
            lambda: (examples["simple"]["prompts"], examples["simple"]["zoom_factors"]),
            outputs=[prompts_input, zoom_factors_input]
        )
        
        # 底部说明
        gr.HTML("""
        <div style="margin-top: 2rem; padding: 1rem; background-color: #f0f0f0; border-radius: 8px;">
            <h3>📖 使用说明</h3>
            <ul>
                <li><strong>文本提示</strong>：输入描述不同尺度的文本，从最大尺度到最小尺度</li>
                <li><strong>缩放因子</strong>：必须是2的幂序列，如 1, 2, 4, 8</li>
                <li><strong>生成</strong>：点击生成按钮创建缩放栈（约需30-60秒）</li>
                <li><strong>浏览</strong>：使用滑块实时查看不同缩放级别的图像</li>
                <li><strong>动画</strong>：创建从粗到细的缩放动画GIF文件</li>
            </ul>
            
            <h3>🎨 支持的图案关键词</h3>
            <p><strong>宇宙系列</strong>：galaxy, spiral, cosmic, space, star, system, planet</p>
            <p><strong>自然系列</strong>：surface, mountain, terrain, tree, forest, branch, bark, insect</p>
            <p><strong>海洋系列</strong>：ocean, water, wave, coral, reef, underwater, fish, tropical</p>
            
            <h3>⚠️ 注意事项</h3>
            <p>• 这是演示版本，使用程序化图案生成，不需要下载AI模型</p>
            <p>• 支持CPU和GPU运行，首次使用时可能需要安装依赖包</p>
            <p>• 生成的图像基于数学函数，提供"Generative Powers of Ten"算法的概念演示</p>
        </div>
        """)
    
    return demo

# ==================== 主函数 ====================

def main():
    """主函数"""
    print("=== Generative Powers of Ten - Gradio Interface ===")
    print(f"设备: {device}")
    print(f"PyTorch版本: {torch.__version__}")
    
    # 检查依赖
    try:
        import gradio
        print(f"Gradio版本: {gradio.__version__}")
    except ImportError:
        print("❌ 错误：请安装Gradio - pip install gradio")
        return
    
    print("✅ 准备就绪")
    print("🎯 演示模式：使用程序化图案生成，无需AI模型")
    
    # 创建并启动界面
    demo = create_gradio_interface()
    
    print("\n🚀 启动Gradio服务器...")
    print("   - 本地访问: http://localhost:7860")
    print("   - 按 Ctrl+C 停止服务器")
    
    # 启动服务器
    demo.launch(
        server_name="0.0.0.0",  # 允许外部访问
        server_port=7860,       # 端口号
        share=False,            # 设为True可创建临时公共URL
        debug=False,            # 调试模式
        show_error=True,        # 显示错误信息
        quiet=False             # 显示启动信息
    )

if __name__ == "__main__":
    main() 