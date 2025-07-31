import torch
from realesrgan import RealESRGANer
from basicsr.archs.rrdbnet_arch import RRDBNet
from PIL import Image

class ESRGANEnhancer:
    def __init__(self, model_path, scale=4):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        # 定义 RRDBNet 模型（ESRGAN 核心）
        self.model = RRDBNet(
            num_in_ch=3, num_out_ch=3, num_feat=64,
            num_block=23, num_grow_ch=32, scale=scale
        )
        # 初始化 RealESRGAN 处理器
        self.upsampler = RealESRGANer(
            scale=scale,
            model_path=model_path,
            model=self.model,
            tile=0,
            tile_pad=10,
            pre_pad=0,
            half=True if self.device == 'cuda' else False,
            device=self.device
        )
        
    def enhance(self, image_path):
        """增强输入图像分辨率"""
        img = Image.open(image_path).convert('RGB')
        img = torch.from_numpy(np.array(img)).permute(2, 0, 1).float() / 255.0
        img = img.unsqueeze(0).to(self.device)
        
        # 超分处理
        output, _ = self.upsampler.enhance(img, outscale=4)
        
        # 转换为 PIL 图像返回
        output = output.squeeze(0).permute(1, 2, 0).cpu().detach().numpy()
        output = (output * 255.0).clip(0, 255).astype(np.uint8)
        return Image.fromarray(output)
    