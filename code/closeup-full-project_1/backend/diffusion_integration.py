import torch
from torchvision import transforms
from PIL import Image
from diffusers import StableDiffusionPipeline, EulerAncestralDiscreteScheduler
import numpy as np

class See3DWithDiffusion:
    """将Diffusion模型与See3D框架结合，实现从文本生成3D模型"""
    
    def __init__(self, diffusion_model_name="stable-diffusion-v1-5", device=None):
        """
        初始化函数
        diffusion_model_name: Diffusion模型名称
        device: 运行设备，默认为自动选择
        """
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.diffusion_model = self._load_diffusion_model(diffusion_model_name)
        self.preprocessor = self._create_preprocessor()
        
    def _load_diffusion_model(self, model_name):
        """加载预训练的Diffusion模型"""
        print(f"Loading diffusion model: {model_name} on {self.device}")
        
        # 加载模型
        pipe = StableDiffusionPipeline.from_pretrained(
            model_name,
            torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
            safety_checker=None
        )
        
        # 设置调度器以获得更好的结果
        pipe.scheduler = EulerAncestralDiscreteScheduler.from_config(pipe.scheduler.config)
        
        # 移动到设备
        pipe = pipe.to(self.device)
        
        # 优化推理
        if self.device == "cuda":
            pipe.enable_xformers_memory_efficient_attention()
            pipe.enable_attention_slicing()
            
        return pipe
    
    def _create_preprocessor(self):
        """创建图像预处理管道"""
        return transforms.Compose([
            transforms.Resize((1024, 768)),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5])
        ])
    
    def generate_image_from_text(self, prompt, negative_prompt="low quality, blurry, distorted",
                                num_inference_steps=50, guidance_scale=7.5, seed=None):
        """
        从文本生成图像
        prompt: 文本提示
        negative_prompt: 负面提示，用于避免生成不想要的特征
        num_inference_steps: 推理步数
        guidance_scale: 引导尺度，控制与提示的匹配程度
        seed: 随机种子，用于结果复现
        返回: 生成的图像和相关参数
        """
        # 设置随机种子
        if seed is not None:
            generator = torch.Generator(device=self.device).manual_seed(seed)
        else:
            generator = None
        
        # 生成图像
        with torch.no_grad():
            result = self.diffusion_model(
                prompt=prompt,
                negative_prompt=negative_prompt,
                num_inference_steps=num_inference_steps,
                guidance_scale=guidance_scale,
                generator=generator
            )
        
        image = result.images[0]
        
        return {
            "image": image,
            "parameters": {
                "prompt": prompt,
                "negative_prompt": negative_prompt,
                "num_inference_steps": num_inference_steps,
                "guidance_scale": guidance_scale,
                "seed": seed,
                "resolution": (image.width, image.height)
            }
        }
    
    def process_image_for_3d(self, image):
        """
        预处理图像以输入到3D模型
        image: PIL图像
        返回: 预处理后的张量
        """
        processed_image = self.preprocessor(image).unsqueeze(0).to(self.device)
        return processed_image
    
    def generate_3d_from_text(self, closeup_gs_model, prompt, **kwargs):
        """
        从文本生成3D模型
        closeup_gs_model: CloseupGS模型实例
        prompt: 文本提示
        **kwargs: 传递给generate_image_from_text的参数
        返回: 生成的图像、3D点云和参数
        """
        # 1. 从文本生成图像
        image_result = self.generate_image_from_text(prompt,** kwargs)
        image = image_result["image"]
        
        # 2. 预处理图像
        processed_image = self.process_image_for_3d(image)
        
        # 3. 生成3D点云
        with torch.no_grad():
            point_cloud = closeup_gs_model(processed_image)
        
        return {
            "image": image,
            "point_cloud": point_cloud,
            "parameters": image_result["parameters"]
        }
    
    def save_generated_image(self, image, save_path):
        """保存生成的图像"""
        image.save(save_path)
        print(f"Generated image saved to {save_path}")
    
    def generate_multiple_views(self, prompt, num_views=3, angles=[0, 45, 90], **kwargs):
        """
        生成同一物体的多个视角图像
        prompt: 基础文本提示
        num_views: 视角数量
        angles: 每个视角的角度描述
        返回: 多个视角的图像列表
        """
        views = []
        
        for i, angle in enumerate(angles[:num_views]):
            # 为每个视角构建提示
            view_prompt = f"{prompt}, viewed from {angle} degrees, professional photography"
            
            # 生成该视角的图像
            result = self.generate_image_from_text(view_prompt,** kwargs)
            views.append({
                "image": result["image"],
                "angle": angle,
                "parameters": result["parameters"]
            })
            
            print(f"Generated view {i+1}/{num_views} at {angle} degrees")
        
        return views
