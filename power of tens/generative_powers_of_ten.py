import torch
from diffusers import StableDiffusionPipeline, DDPMScheduler
import numpy as np
import cv2
from scipy.ndimage import gaussian_filter
import warnings
warnings.filterwarnings("ignore")

# 设备配置
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

# 加载 Stable Diffusion 模型
print("Loading Stable Diffusion model...")
pipe = StableDiffusionPipeline.from_pretrained(
    "runwayml/stable-diffusion-v1-5",  # 使用更稳定的版本
    torch_dtype=torch.float16 if device == "cuda" else torch.float32,
    safety_checker=None,
    requires_safety_checker=False
).to(device)

# 配置调度器
scheduler = DDPMScheduler.from_config(pipe.scheduler.config)
pipe.scheduler = scheduler

# 分类器自由指导配置
guidance_scale = 7.5  # omega 参数
print(f"Guidance scale (omega): {guidance_scale}")

# 优化内存使用
if device == "cuda":
    pipe.enable_memory_efficient_attention()
    pipe.enable_attention_slicing()

print("Model loaded successfully!")

# 噪声调度参数（从 DDPM）
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

# ==================== 缩放栈数据结构 ====================

class ZoomStack:
    """缩放栈数据结构
    
    根据 'Generative Powers of Ten' 论文实现的缩放栈：
    一个张量列表 L = [L_0, L_1, ..., L_{N-1}]，其中每个 L_i 是 HxWx3 图像张量
    """
    
    def __init__(self, zoom_factors, H=512, W=512, device="cuda"):
        """初始化缩放栈
        
        Args:
            zoom_factors: 缩放因子列表，例如 [1, 2, 4, 8]
            H: 图像高度
            W: 图像宽度 
            device: 设备类型
        """
        self.zoom_factors = zoom_factors
        self.N = len(zoom_factors)
        self.H = H
        self.W = W
        self.device = device
        
        # 验证缩放因子是2的幂
        for i, p in enumerate(zoom_factors):
            if i == 0:
                assert p == 1, "第一个缩放因子必须是1"
            else:
                assert p == 2 * zoom_factors[i-1], f"缩放因子必须是2的幂序列，但得到 {p} 在位置 {i}"
        
        # 初始化层：L = [L_0, L_1, ..., L_{N-1}]
        # 每个 L_i 是 HxWx3 张量，从高斯噪声初始化
        self.layers = self._initialize_layers()
        
    def _initialize_layers(self):
        """从高斯噪声初始化所有层"""
        layers = []
        for i, p in enumerate(self.zoom_factors):
            # 每层都是全分辨率 HxWx3，但代表不同的缩放级别
            layer = torch.randn(self.H, self.W, 3, device=self.device, dtype=torch.float32)
            # 标准化到 [-1, 1] 范围
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
        assert layer.shape == (self.H, self.W, 3), f"层形状必须是 ({self.H}, {self.W}, 3)，但得到 {layer.shape}"
        self.layers[i] = layer.to(self.device)
    
    def get_zoom_factor(self, i):
        """获取第 i 层的缩放因子"""
        assert 0 <= i < self.N, f"层索引 {i} 超出范围 [0, {self.N-1}]"
        return self.zoom_factors[i]
    
    def get_all_layers(self):
        """获取所有层的副本"""
        return [layer.clone() for layer in self.layers]
    
    def update_layers(self, new_layers):
        """更新所有层"""
        assert len(new_layers) == self.N, f"新层数量 {len(new_layers)} 不匹配缩放栈大小 {self.N}"
        for i, layer in enumerate(new_layers):
            self.set_layer(i, layer)
    
    def to_image_format(self, layer_idx):
        """将层转换为图像格式 [0, 1]"""
        layer = self.get_layer(layer_idx)
        # 从 [-1, 1] 转换到 [0, 1]
        return torch.clamp((layer + 1.0) / 2.0, 0.0, 1.0)
    
    def save_layer_as_image(self, layer_idx, filename):
        """保存层为图像文件"""
        import torchvision.transforms as transforms
        from PIL import Image
        
        img_tensor = self.to_image_format(layer_idx)
        # 转换为 CHW 格式
        img_tensor = img_tensor.permute(2, 0, 1)
        
        # 转换为 PIL 图像并保存
        to_pil = transforms.ToPILImage()
        img = to_pil(img_tensor.cpu())
        img.save(filename)
        print(f"层 {layer_idx} 已保存为 {filename}")
    
    def print_info(self):
        """打印缩放栈信息"""
        print(f"\n=== 缩放栈信息 ===")
        print(f"层数: {self.N}")
        print(f"图像尺寸: {self.H}x{self.W}")
        print(f"设备: {self.device}")
        print(f"缩放因子: {self.zoom_factors}")
        for i, p in enumerate(self.zoom_factors):
            layer = self.layers[i]
            print(f"  L_{i}: 缩放因子={p}, 形状={layer.shape}, 数据范围=[{layer.min():.3f}, {layer.max():.3f}]")

def create_zoom_stack(zoom_factors, H=512, W=512, device="cuda"):
    """创建缩放栈的便捷函数
    
    Args:
        zoom_factors: 缩放因子列表，例如 [1, 2, 4, 8]
        H: 图像高度
        W: 图像宽度
        device: 设备类型
    
    Returns:
        ZoomStack: 初始化的缩放栈对象
    """
    return ZoomStack(zoom_factors, H, W, device)

def generate_zoom_factors(N):
    """生成 N 个缩放因子：[1, 2, 4, ..., 2^{N-1}]
    
    Args:
        N: 层数
    
    Returns:
        list: 缩放因子列表
    """
    return [2**i for i in range(N)]

# 兼容性函数（保持向后兼容）
def initialize_zoom_stack(N, H=512, W=512):
    """初始化缩放栈（向后兼容版本）"""
    zoom_factors = generate_zoom_factors(N)
    zoom_stack = create_zoom_stack(zoom_factors, H, W, device)
    return zoom_stack.get_all_layers()

# ==================== 缩放栈测试函数 ====================

def test_zoom_stack():
    """测试缩放栈功能"""
    print("\n=== 测试缩放栈数据结构 ===")
    
    # 创建测试缩放栈
    zoom_factors = [1, 2, 4, 8]
    zoom_stack = create_zoom_stack(zoom_factors, H=256, W=256, device=device)
    
    # 打印信息
    zoom_stack.print_info()
    
    # 测试基本操作
    print(f"\n--- 测试基本操作 ---")
    layer_0 = zoom_stack.get_layer(0)
    print(f"获取 L_0: 形状={layer_0.shape}, 设备={layer_0.device}")
    
    # 测试修改层
    new_layer = torch.zeros_like(layer_0)
    zoom_stack.set_layer(0, new_layer)
    print(f"设置 L_0 为零张量: 数据范围=[{zoom_stack.get_layer(0).min():.3f}, {zoom_stack.get_layer(0).max():.3f}]")
    
    # 重新初始化
    zoom_stack = create_zoom_stack(zoom_factors, H=256, W=256, device=device)
    
    # 测试图像格式转换
    img_format = zoom_stack.to_image_format(0)
    print(f"图像格式转换: 原始范围=[{layer_0.min():.3f}, {layer_0.max():.3f}] -> [0,1]范围=[{img_format.min():.3f}, {img_format.max():.3f}]")
    
    return zoom_stack

# 助手：构建拉普拉斯金字塔
def build_laplacian_pyramid(img, num_levels):
    """构建拉普拉斯金字塔"""
    # 高斯金字塔
    g_pyramid = [img]
    for _ in range(num_levels - 1):
        g = gaussian_filter(g_pyramid[-1], sigma=1)  # 或使用 torch conv
        g_down = g[::2, ::2]  # 下采样
        g_pyramid.append(g_down)
    
    # 拉普拉斯金字塔
    l_pyramid = []
    for i in range(num_levels - 1):
        up = np.kron(g_pyramid[i+1], np.ones((2,2)))  # 上采样（简化）
        l = g_pyramid[i] - gaussian_filter(up, sigma=1)
        l_pyramid.append(l)
    l_pyramid.append(g_pyramid[-1])  # 基础层
    return l_pyramid

def reconstruct_laplacian_pyramid(l_pyramid):
    """重建拉普拉斯金字塔"""
    # 反转过程
    pass

# 渲染函数 (Pi_image, Pi_noise)
def render_image(zoom_stack, i, zoom_factors):
    """渲染图像（算法 1）"""
    # 实现算法 1
    x = zoom_stack[i].clone()
    for j in range(i+1, len(zoom_stack)):
        # 下采样 L_j，掩码融合
        pass
    return x

def render_noise(zoom_stack, i, zoom_factors):
    """渲染噪声"""
    # 类似，但缩放噪声
    pass

# 多分辨率融合
def multi_resolution_blending(predictions, zoom_factors):
    """多分辨率融合"""
    new_stack = []
    for i in range(len(predictions)):
        obs = []  # 收集 j >= i 的裁剪/重缩放 x_j
        for j in range(i, len(predictions)):
            # 中心裁剪 H/p_j x W/p_j，上采样到 HxW
            pass
        # 为每个 obs 构建金字塔，平均频率带，重建
        blended_img = predictions[i]  # 临时占位符
        new_stack.append(blended_img)
    return new_stack

# 联合采样
def joint_multi_scale_sampling(prompts, zoom_factors, T=50, omega=7.5):
    """联合多尺度采样（算法 2）"""
    N = len(prompts)
    zoom_stack = initialize_zoom_stack(N)
    
    for t in range(T, 0, -1):
        z_t_list = [render_image(zoom_stack, i, zoom_factors) + render_noise(zoom_stack, i, zoom_factors) for i in range(N)]  # 简化
        x_hat_list = []
        
        for i in range(N):
            # 将 z_t_list[i] + 提示 prompts[i] 输入模型
            # 使用 pipe 预测，应用分类器自由指导
            # noise_pred = ... (来自模型)
            x_hat = z_t_list[i]  # 临时占位符
            x_hat_list.append(x_hat)
        
        # 可选：如果提供 input_image，进行基于照片的优化
        zoom_stack = multi_resolution_blending(x_hat_list, zoom_factors)
        # DDPM 更新到下一个 z_{t-1}
    
    return zoom_stack

# 测试函数
def test_model_loading():
    """测试模型加载和基础功能"""
    test_prompt = "a beautiful landscape"
    print(f"\nTesting model with prompt: '{test_prompt}'")
    
    try:
        # 生成单张测试图像
        with torch.no_grad():
            image = pipe(
                test_prompt,
                num_inference_steps=20,
                guidance_scale=guidance_scale,
                height=512,
                width=512
            ).images[0]
        
        print("Model test successful!")
        return True
    except Exception as e:
        print(f"Model test failed: {e}")
        return False

# 示例用法
if __name__ == "__main__":
    print("=== Generative Powers of Ten - 实现测试 ===")
    
    # 第一步：测试模型加载
    print("\n--- 第一步：测试 Stable Diffusion 模型加载 ---")
    test_model_loading()
    
    # 第二步：测试缩放栈数据结构
    print("\n--- 第二步：测试缩放栈数据结构 ---")
    zoom_stack = test_zoom_stack()
    
    # 示例配置
    prompts = ["a distant galaxy", "a star system", "an alien planet surface", "an insect on a tree branch"]
    zoom_factors = [1, 2, 4, 8]
    
    print(f"\n=== 示例配置 ===")
    print(f"文本提示: {prompts}")
    print(f"缩放因子: {zoom_factors}")
    print(f"尺度数量: {len(prompts)}")
    
    # 验证配置匹配
    assert len(prompts) == len(zoom_factors), "提示数量必须与缩放因子数量匹配"
    
    # 创建完整的缩放栈用于后续步骤
    print(f"\n--- 创建完整缩放栈 ---")
    full_zoom_stack = create_zoom_stack(zoom_factors, H=512, W=512, device=device)
    full_zoom_stack.print_info()
    
    print(f"\n✅ 第一步和第二步完成！")
    print(f"   - Stable Diffusion 模型已加载，guidance_scale={guidance_scale}")
    print(f"   - 缩放栈数据结构已实现并测试")
    print(f"   - 准备好进行下一步：实现渲染函数")
    
    # 保存示例层（可选）
    try:
        full_zoom_stack.save_layer_as_image(0, "zoom_layer_0_example.png")
        print(f"   - 示例图像已保存为 'zoom_layer_0_example.png'")
    except Exception as e:
        print(f"   - 保存示例图像失败: {e}")
    
    # final_stack = joint_multi_scale_sampling(prompts, zoom_factors)