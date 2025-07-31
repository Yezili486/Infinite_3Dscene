import torch
from zoedepth.models.builder import build_model
from zoedepth.utils.config import get_config
from PIL import Image
import numpy as np

class ZoeDepthEstimator:
    def __init__(self, model_path):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        # 加载 ZoeDepth 配置和模型
        self.config = get_config("zoedepth", "infer")
        self.model = build_model(self.config).to(self.device)
        self.model.load_state_dict(torch.load(model_path, map_location=self.device))
        self.model.eval()
        
    def estimate(self, image):
        """从输入图像估计深度图"""
        # 图像预处理
        img = image.resize((640, 480))  # 调整为模型输入尺寸
        img = np.array(img) / 255.0
        img = torch.from_numpy(img).permute(2, 0, 1).float().unsqueeze(0).to(self.device)
        
        # 深度估计
        with torch.no_grad():
            depth = self.model(img)['metric_depth']  # 获取 metric 深度
        
        # 后处理：归一化到 [0, 255] 便于可视化
        depth = depth.squeeze().cpu().numpy()
        depth = (depth - depth.min()) / (depth.max() - depth.min()) * 255.0
        return depth.astype(np.uint8)
    