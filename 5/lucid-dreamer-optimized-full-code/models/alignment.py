import torch
import torch.nn.functional as F
from zoedepth.models.builder import build_model
from zoedepth.utils.config import get_config

class DepthEnhancedAlignment:
    def __init__(self, config, gaussian_model):
        self.config = config
        self.device = config['device']
        self.gaussian_model = gaussian_model  # 3DGS模型
        self.depth_loss_weight = config['model']['depth_loss_weight']
        
        # 初始化ZoeDepth模型
        self.zoe = self._init_zoe_depth()

    def _init_zoe_depth(self):
        """初始化ZoeDepth深度估计模型"""
        config_path = "zoedepth_nk"  # 可根据需要更换模型配置
        cfg = get_config(config_path)
        model = build_model(cfg)
        model.load_state_dict(torch.load(self.config['paths']['zoe_model'], map_location=self.device))
        model.to(self.device).eval()
        return model

    def estimate_depth(self, rgb_image):
        """估计RGB图像的深度信息"""
        # 调整图像尺寸以匹配ZoeDepth输入要求
        input_img = F.interpolate(rgb_image, size=(384, 512), mode='bilinear', align_corners=False)
        with torch.no_grad():
            depth = self.zoe(input_img)['metric_depth']
        # 恢复到原始尺寸
        depth = F.interpolate(depth, size=rgb_image.shape[2:], mode='bilinear', align_corners=False)
        return depth

    def render_with_depth(self, camera_params):
        """渲染图像并返回深度信息"""
        # 渲染RGB图像和深度图
        rendered_img, rendered_depth = self.gaussian_model.render(camera_params, return_depth=True)
        return rendered_img, rendered_depth

    def compute_alignment_loss(self, target_img, camera_params):
        """计算对齐损失，包含像素损失和深度损失"""
        # 渲染当前视角的图像和深度
        rendered_img, rendered_depth = self.render_with_depth(camera_params)
        
        # 1. 像素损失
        pixel_loss = F.l1_loss(rendered_img, target_img)
        
        # 2. 深度损失
        target_depth = self.estimate_depth(target_img)
        depth_loss = F.smooth_l1_loss(rendered_depth, target_depth)
        
        # 总损失
        total_loss = pixel_loss + self.depth_loss_weight * depth_loss
        return total_loss, {
            'total_loss': total_loss.item(),
            'pixel_loss': pixel_loss.item(),
            'depth_loss': depth_loss.item()
        }
