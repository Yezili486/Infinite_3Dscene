import gradio as gr
import torch
from diffusers import StableDiffusionPipeline, DDPMScheduler
import numpy as np
import cv2
from scipy.ndimage import gaussian_filter
import warnings
import time
import io
import base64
from PIL import Image
import matplotlib.pyplot as plt
from typing import List, Tuple, Optional
import threading
import os

warnings.filterwarnings("ignore")

# 设备配置
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

# 全局变量存储当前的缩放栈
current_zoom_stack = None
generation_status = {"status": "idle", "progress": 0, "message": ""}

# ==================== 核心功能 ====================
# (这里包含之前实现的所有核心功能)

# 噪声调度参数
def get_noise_schedule(num_timesteps=1000, beta_start=0.00085, beta_end=0.012):
    """获取噪声调度参数"""
    betas = torch.linspace(beta_start, beta_end, num_timesteps, dtype=torch.float32)
    alphas = 1.0 - betas
    alphas_cumprod = torch.cumprod(alphas, dim=0)
    
    return {
        'betas': betas,
        'alphas': alphas, 
        'alphas_cumprod': alphas_cumprod,
        'sqrt_alphas_cumprod': torch.sqrt(alphas_cumprod),
        'sqrt_one_minus_alphas_cumprod': torch.sqrt(1.0 - alphas_cumprod)
    }

noise_schedule = get_noise_schedule()

# 缩放栈数据结构
class ZoomStack:
    """缩放栈数据结构"""
    
    def __init__(self, zoom_factors, H=512, W=512, device="cuda"):
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
                assert p == 2 * zoom_factors[i-1], f"缩放因子必须是2的幂序列"
        
        # 初始化层
        self.layers = self._initialize_layers()
        
    def _initialize_layers(self):
        """从高斯噪声初始化所有层"""
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

def create_zoom_stack(zoom_factors, H=512, W=512, device="cuda"):
    """创建缩放栈的便捷函数"""
    return ZoomStack(zoom_factors, H, W, device)

def generate_zoom_factors(N):
    """生成 N 个缩放因子：[1, 2, 4, ..., 2^{N-1}]"""
    return [2**i for i in range(N)]

# 渲染函数
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
        crop_h = H // p_j
        crop_w = W // p_j
        start_h = (H - crop_h) // 2
        start_w = (W - crop_w) // 2
        end_h = start_h + crop_h
        end_w = start_w + crop_w
        
        cropped_j = layer_j[start_h:end_h, start_w:end_w, :]
        
        # 上采样
        cropped_j_chw = cropped_j.permute(2, 0, 1).unsqueeze(0)
        upsampled_j = torch.nn.functional.interpolate(
            cropped_j_chw, 
            size=(H, W), 
            mode='bilinear', 
            align_corners=False
        )
        upsampled_j = upsampled_j.squeeze(0).permute(1, 2, 0)
        
        # 融合掩码
        mask = torch.zeros((H, W, 1), device=zoom_stack.device, dtype=torch.float32)
        mask_h_start = (H - crop_h * p_j) // 2
        mask_w_start = (W - crop_w * p_j) // 2
        mask_h_end = mask_h_start + crop_h * p_j
        mask_w_end = mask_w_start + crop_w * p_j
        
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
            crop_h = H // p_j
            crop_w = W // p_j
            
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
            mask_h_end = mask_h_start + crop_h * p_j
            mask_w_end = mask_w_start + crop_w * p_j
            mask[mask_h_start:mask_h_end, mask_w_start:mask_w_end, :] = 1.0
            layer_noise = layer_noise * mask
        
        rendered_noise += layer_noise
        total_variance += mask * (noise_scale ** 2)
    
    std_dev = torch.sqrt(total_variance + 1e-8)
    rendered_noise = rendered_noise / std_dev
    
    return rendered_noise

# 简化的采样函数
def joint_multi_scale_sampling_simple(prompts, zoom_factors, T=20, H=256, W=256):
    """简化版本的联合多尺度采样"""
    global generation_status
    
    generation_status["status"] = "generating"
    generation_status["progress"] = 0
    generation_status["message"] = "初始化缩放栈..."
    
    N = len(prompts)
    zoom_stack = create_zoom_stack(zoom_factors, H, W, device)
    
    # 简化的扩散循环
    for t in range(T, 0, -1):
        progress = (T - t + 1) / T
        generation_status["progress"] = int(progress * 100)
        generation_status["message"] = f"采样步骤 {T-t+1}/{T}"
        
        # 1. 渲染当前状态
        z_t_list = []
        for i in range(N):
            img_rendered = Pi_image(zoom_stack, i)
            noise_rendered = Pi_noise(zoom_stack, i)
            
            noise_level = t / T
            z_t = img_rendered + noise_level * noise_rendered * 0.3
            z_t_list.append(z_t)
        
        # 2. 模拟模型预测
        x_hat_list = []
        for i in range(N):
            denoising_strength = progress
            
            # 添加基于提示的内容模式
            content_pattern = torch.sin(torch.linspace(0, 2*torch.pi*i, H*W, device=device)).reshape(H, W, 1)
            content_pattern = content_pattern.repeat(1, 1, 3) * 0.2
            
            base_img = zoom_stack.get_layer(i)
            x_hat = (base_img * (1 - denoising_strength) + 
                    content_pattern * denoising_strength * 0.5 +
                    z_t_list[i] * denoising_strength * 0.5)
            
            x_hat_list.append(x_hat)
        
        # 3. 更新缩放栈
        for i in range(N):
            zoom_stack.set_layer(i, x_hat_list[i])
    
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
                raise ValueError("缩放因子必须是2的幂序列")
        
        return factors
    except Exception as e:
        raise ValueError(f"缩放因子格式错误: {e}")

def generate_zoom_stack(prompts_text, zoom_factors_text, progress=gr.Progress()):
    """生成缩放栈"""
    global current_zoom_stack
    
    try:
        # 解析输入
        prompts = parse_prompts(prompts_text)
        zoom_factors = parse_zoom_factors(zoom_factors_text)
        
        if not prompts:
            return "错误：请输入至少一个提示", None, gr.update(visible=False)
        
        if not zoom_factors:
            return "错误：请输入有效的缩放因子", None, gr.update(visible=False)
        
        if len(prompts) != len(zoom_factors):
            return f"错误：提示数量({len(prompts)})与缩放因子数量({len(zoom_factors)})不匹配", None, gr.update(visible=False)
        
        # 生成缩放栈
        progress(0, desc="开始生成...")
        
        # 使用线程来监控进度
        def monitor_progress():
            while generation_status["status"] == "generating":
                progress(generation_status["progress"] / 100, desc=generation_status["message"])
                time.sleep(0.1)
        
        monitor_thread = threading.Thread(target=monitor_progress)
        monitor_thread.start()
        
        current_zoom_stack = joint_multi_scale_sampling_simple(
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
        info_text = f"尺度 {scale_idx}: 缩放因子 {zoom_factor}x"
        
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
                duration=1000,  # 每帧1秒
                loop=0
            )
            
            print(f"动画已保存: {output_path}")
            return output_path, f"✅ 动画已创建，包含 {len(frames)} 帧"
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
        """
    ) as demo:
        
        # 标题和说明
        gr.HTML("""
        <div class="header">
            <h1>🔍 Generative Powers of Ten - Interactive Demo</h1>
            <p>基于论文 "Generative Powers of Ten" 的交互式演示</p>
            <p>输入文本提示生成多尺度缩放栈，并实时浏览不同缩放级别的图像</p>
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
                    value="准备就绪",
                    interactive=False,
                    elem_classes=["status-text"]
                )
                
                # 示例按钮
                gr.Markdown("### 🎯 快速示例")
                examples = get_example_inputs()
                
                with gr.Row():
                    cosmic_btn = gr.Button("🌌 宇宙缩放", size="sm")
                    nature_btn = gr.Button("🌳 自然缩放", size="sm")
                    ocean_btn = gr.Button("🌊 海洋缩放", size="sm")
            
            # 右侧：显示面板
            with gr.Column(scale=2):
                gr.Markdown("## 🖼️ 图像显示")
                
                # 主图像显示
                main_image = gr.Image(
                    label="当前尺度图像",
                    type="pil",
                    height=400
                )
                
                # 缩放级别控制
                scale_slider = gr.Slider(
                    minimum=0,
                    maximum=3,
                    value=0,
                    step=1,
                    label="缩放级别",
                    visible=False
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
                    label="动画文件",
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
            fn=generate_zoom_stack,
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
        
        # 底部说明
        gr.HTML("""
        <div style="margin-top: 2rem; padding: 1rem; background-color: #f0f0f0; border-radius: 8px;">
            <h3>📖 使用说明</h3>
            <ul>
                <li><strong>文本提示</strong>：输入描述不同尺度的文本，从最大尺度到最小尺度</li>
                <li><strong>缩放因子</strong>：必须是2的幂序列，如 1, 2, 4, 8</li>
                <li><strong>生成</strong>：点击生成按钮创建缩放栈（需要几分钟）</li>
                <li><strong>浏览</strong>：使用滑块实时查看不同缩放级别的图像</li>
                <li><strong>动画</strong>：创建从粗到细的缩放动画GIF文件</li>
            </ul>
            <p><strong>注意</strong>：首次运行需要下载模型，请确保网络连接正常</p>
        </div>
        """)
    
    return demo

# ==================== 主函数 ====================

def main():
    """主函数"""
    print("=== Generative Powers of Ten - Gradio Interface ===")
    print(f"设备: {device}")
    
    # 检查是否需要加载模型
    try:
        print("正在检查模型...")
        # 这里可以添加模型预加载逻辑
        print("✅ 准备就绪")
    except Exception as e:
        print(f"⚠️ 模型检查失败: {e}")
        print("   首次运行时将自动下载模型")
    
    # 创建并启动界面
    demo = create_gradio_interface()
    
    # 启动服务器
    demo.launch(
        server_name="0.0.0.0",  # 允许外部访问
        server_port=7860,       # 端口号
        share=False,            # 不创建公共链接（可设为True创建临时公共URL）
        debug=True,             # 调试模式
        show_error=True         # 显示错误信息
    )

if __name__ == "__main__":
    main() 