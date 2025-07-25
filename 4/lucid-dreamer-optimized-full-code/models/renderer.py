import torch
import numpy as np
from math import pi
from typing import Tuple, Dict, List

class GaussianRenderer:
    def __init__(self, config):
        self.config = config
        self.device = config['device']
        self.background = torch.tensor(config['model']['renderer']['background_color'], device=self.device, dtype=torch.float32)
        self.num_samples = config['model']['renderer']['num_samples']
        
    def render(self, gaussians, camera_params, return_depth=False) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        渲染3D高斯球场景
        
        参数:
            gaussians: 高斯球模型参数
            camera_params: 相机参数
            return_depth: 是否返回深度图
            
        返回:
            rendered_img: 渲染的RGB图像
            rendered_depth: 深度图（如果return_depth=True）
        """
        # 解析相机参数
        cam_pos = torch.tensor(camera_params['position'], device=self.device, dtype=torch.float32)
        cam_rot = torch.tensor(camera_params['rotation'], device=self.device, dtype=torch.float32)
        focal = camera_params['focal']
        cx, cy = camera_params['cx'], camera_params['cy']
        width = int(camera_params.get('width', self.config['model']['progressive_training']['stage2_res']))
        height = int(camera_params.get('height', self.config['model']['progressive_training']['stage2_res']))
        
        # 生成射线
        rays_o, rays_d = self._generate_rays(cam_pos, cam_rot, focal, cx, cy, width, height)
        
        # 渲染射线
        rgb, depth = self._render_rays(gaussians, rays_o, rays_d)
        
        # 调整形状为图像格式
        rgb = rgb.reshape(height, width, 3).permute(2, 0, 1)  # (3, H, W)
        depth = depth.reshape(height, width, 1).permute(2, 0, 1)  # (1, H, W)
        
        if return_depth:
            return rgb, depth
        return rgb
    
    def _generate_rays(self, cam_pos, cam_rot, focal, cx, cy, width, height) -> Tuple[torch.Tensor, torch.Tensor]:
        """生成相机射线"""
        # 像素坐标
        i, j = torch.meshgrid(
            torch.arange(width, device=self.device),
            torch.arange(height, device=self.device),
            indexing='xy'
        )
        i = i.t().float()  # (H, W)
        j = j.t().float()  # (H, W)
        
        # 射线方向（相机坐标系）
        dirs = torch.stack([(i - cx) / focal, -(j - cy) / focal, -torch.ones_like(i)], dim=-1)  # (H, W, 3)
        
        # 旋转矩阵（欧拉角转旋转矩阵）
        rot_mat = self._euler_to_rotmat(cam_rot)
        
        # 转换到世界坐标系
        rays_d = torch.sum(dirs[..., None, :] * rot_mat, dim=-1)  # (H, W, 3)
        rays_d = rays_d / torch.norm(rays_d, dim=-1, keepdim=True)  # 归一化
        
        # 射线原点（相机位置）
        rays_o = cam_pos.expand(rays_d.shape)  # (H, W, 3)
        
        return rays_o.reshape(-1, 3), rays_d.reshape(-1, 3)  # (H*W, 3), (H*W, 3)
    
    def _euler_to_rotmat(self, euler: torch.Tensor) -> torch.Tensor:
        """欧拉角转旋转矩阵 (x, y, z) -> 旋转矩阵"""
        x, y, z = euler * pi / 180.0  # 转换为弧度
        
        # 绕x轴旋转
        Rx = torch.tensor([
            [1, 0, 0],
            [0, torch.cos(x), -torch.sin(x)],
            [0, torch.sin(x), torch.cos(x)]
        ], device=self.device)
        
        # 绕y轴旋转
        Ry = torch.tensor([
            [torch.cos(y), 0, torch.sin(y)],
            [0, 1, 0],
            [-torch.sin(y), 0, torch.cos(y)]
        ], device=self.device)
        
        # 绕z轴旋转
        Rz = torch.tensor([
            [torch.cos(z), -torch.sin(z), 0],
            [torch.sin(z), torch.cos(z), 0],
            [0, 0, 1]
        ], device=self.device)
        
        return Rz @ Ry @ Rx  # 组合旋转矩阵
    
    def _render_rays(self, gaussians, rays_o, rays_d) -> Tuple[torch.Tensor, torch.Tensor]:
        """渲染射线，计算颜色和深度"""
        # 简化实现，实际应使用3DGS的光栅化方法
        num_rays = rays_o.shape[0]
        rgb = torch.ones(num_rays, 3, device=self.device) * self.background
        depth = torch.zeros(num_rays, 1, device=self.device)
        
        # 获取高斯球参数
        means = gaussians['means']
        scales = gaussians['scales']
        rotations = gaussians['rotations']
        opacities = gaussians['opacities']
        colors = gaussians['colors']
        
        # 计算每个高斯球对射线的贡献（简化计算）
        for i in range(means.shape[0]):
            mean = means[i]
            scale = scales[i]
            opacity = opacities[i]
            color = colors[i]
            
            # 计算射线到高斯球中心的距离
            t = torch.dot(mean - rays_o[0], rays_d[0])  # 简化为单条射线计算
            if t > 0:
                # 计算贡献（简化模型）
                contribution = opacity * torch.exp(-0.5 * t * t / (scale * scale))
                rgb += contribution * color
                depth += contribution * t
        
        # 归一化
        rgb = torch.clamp(rgb, 0.0, 1.0)
        return rgb, depth

class GaussianModel:
    def __init__(self, config):
        self.config = config
        self.device = config['device']
        self.renderer = GaussianRenderer(config)
        self.render_params = {
            'width': config['model']['progressive_training']['stage2_res'],
            'height': config['model']['progressive_training']['stage2_res']
        }
        
        # 初始化高斯球参数
        self._init_gaussians()
    
    def _init_gaussians(self):
        """初始化高斯球参数"""
        num_points = self.config['model']['gaussian_args']['num_points']
        initial_scale = self.config['model']['gaussian_args']['initial_scale']
        
        # 随机初始化高斯球中心
        self.means = torch.randn(num_points, 3, device=self.device) * 0.5
        # 初始化尺度
        self.scales = torch.ones(num_points, 3, device=self.device) * initial_scale
        # 初始化旋转（单位四元数）
        self.rotations = torch.zeros(num_points, 4, device=self.device)
        self.rotations[:, 0] = 1.0  # 单位四元数 (1, 0, 0, 0)
        # 初始化不透明度
        self.opacities = torch.sigmoid(torch.randn(num_points, 1, device=self.device))
        # 初始化颜色
        self.colors = torch.sigmoid(torch.randn(num_points, 3, device=self.device))
        
        # 标记可训练参数
        self.means.requires_grad = True
        self.scales.requires_grad = True
        self.rotations.requires_grad = True
        self.opacities.requires_grad = True
        self.colors.requires_grad = True
    
    def parameters(self):
        """返回可训练参数"""
        return [self.means, self.scales, self.rotations, self.opacities, self.colors]
    
    def render(self, camera_params, return_depth=False):
        """渲染场景"""
        # 准备高斯球参数
        gaussians = {
            'means': self.means,
            'scales': self.scales,
            'rotations': self.rotations,
            'opacities': self.opacities,
            'colors': self.colors
        }
        
        # 调用渲染器
        return self.renderer.render(gaussians, camera_params, return_depth)
    
    def set_resolution(self, resolution):
        """设置渲染分辨率"""
        self.render_params['width'] = resolution
        self.render_params['height'] = resolution
