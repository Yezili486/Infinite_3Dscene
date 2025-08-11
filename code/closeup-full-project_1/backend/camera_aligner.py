import torch
import torch.nn as nn
import numpy as np
from torchvision import transforms
from PIL import Image
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

class Camera:
    """相机类，定义相机参数和投影变换"""
    
    def __init__(self, fov=60, aspect_ratio=4/3, near=0.1, far=1000, device="cuda"):
        self.fov = fov  # 视场角，单位为度
        self.aspect_ratio = aspect_ratio  # 宽高比
        self.near = near  # 近平面
        self.far = far    # 远平面
        self.device = device
        
        # 初始相机位置和姿态
        self.position = torch.tensor([0.0, 0.0, 5.0], device=device, requires_grad=False)
        self.rotation = torch.tensor([0.0, 0.0, 0.0], device=device, requires_grad=False)  # 欧拉角 (x, y, z)
        
        # 计算投影矩阵
        self.projection_matrix = self._compute_projection_matrix()
    
    def _compute_projection_matrix(self):
        """计算透视投影矩阵"""
        fov_rad = torch.tensor(self.fov, device=self.device) * torch.pi / 180
        tan_half_fov = torch.tan(fov_rad / 2)
        
        proj = torch.zeros((4, 4), device=self.device)
        
        proj[0, 0] = 1 / (tan_half_fov * self.aspect_ratio)
        proj[1, 1] = 1 / tan_half_fov
        proj[2, 2] = -(self.far + self.near) / (self.far - self.near)
        proj[2, 3] = -(2 * self.far * self.near) / (self.far - self.near)
        proj[3, 2] = -1
        
        return proj
    
    def _compute_view_matrix(self):
        """计算视图矩阵（相机姿态）"""
        # 从欧拉角计算旋转矩阵
        rx, ry, rz = self.rotation
        
        # X轴旋转
        Rx = torch.tensor([
            [1, 0, 0, 0],
            [0, torch.cos(rx), -torch.sin(rx), 0],
            [0, torch.sin(rx), torch.cos(rx), 0],
            [0, 0, 0, 1]
        ], device=self.device)
        
        # Y轴旋转
        Ry = torch.tensor([
            [torch.cos(ry), 0, torch.sin(ry), 0],
            [0, 1, 0, 0],
            [-torch.sin(ry), 0, torch.cos(ry), 0],
            [0, 0, 0, 1]
        ], device=self.device)
        
        # Z轴旋转
        Rz = torch.tensor([
            [torch.cos(rz), -torch.sin(rz), 0, 0],
            [torch.sin(rz), torch.cos(rz), 0, 0],
            [0, 0, 1, 0],
            [0, 0, 0, 1]
        ], device=self.device)
        
        # 组合旋转矩阵
        R = Rz @ Ry @ Rx
        
        # 平移矩阵
        T = torch.eye(4, device=self.device)
        T[:3, 3] = -self.position
        
        # 视图矩阵 = 旋转 * 平移
        view_matrix = R @ T
        
        return view_matrix
    
    def project_points(self, points):
        """
        将3D点投影到2D图像平面
        points: 3D点张量，形状为 [N, 3]
        返回: 投影后的2D点，形状为 [N, 2]
        """
        # 转换为齐次坐标 [N, 4]
        homogeneous_points = torch.cat([points, torch.ones(points.shape[0], 1, device=self.device)], dim=1)
        
        # 应用视图变换和投影变换
        view_matrix = self._compute_view_matrix()
        projected = homogeneous_points @ view_matrix.T @ self.projection_matrix.T
        
        # 透视除法
        projected = projected[:, :3] / projected[:, 3:4]
        
        # 返回标准化设备坐标 (NDC)，范围 [-1, 1]
        return projected[:, :2]
    
    def set_pose(self, position=None, rotation=None):
        """设置相机位置和旋转"""
        if position is not None:
            self.position = torch.tensor(position, device=self.device)
        if rotation is not None:
            # 将角度从度转换为弧度
            rotation_rad = torch.tensor(rotation, device=self.device) * torch.pi / 180
            self.rotation = rotation_rad

class CameraViewAligner:
    """相机视角对齐器，用于优化相机参数以匹配参考图像"""
    
    def __init__(self, initial_rotation=[0.0, 0.0, 0.0], device="cuda"):
        self.device = device
        
        # 可优化的相机旋转参数（角度，度）
        self.rotation = torch.tensor(initial_rotation, dtype=torch.float32, 
                                    device=device, requires_grad=True)
        
        # 优化器
        self.optimizer = torch.optim.Adam([self.rotation], lr=0.5)
        
        # 损失函数
        self.loss_fn = nn.MSELoss()
        
        # 记录最佳结果
        self.best_loss = float('inf')
        self.best_rotation = initial_rotation.copy()
        
        # 记录损失历史
        self.loss_history = []
    
    def _preprocess_image(self, image):
        """预处理参考图像"""
        preprocess = transforms.Compose([
            transforms.Resize((512, 512)),
            transforms.ToTensor()
        ])
        return preprocess(image).to(self.device)
    
    def align(self, renderer, point_cloud, reference_image, iterations=20, verbose=True):
        """
        渐进式相机视角对齐
        renderer: 渲染器实例
        point_cloud: 3D点云数据
        reference_image: 参考图像（PIL Image）
        iterations: 优化迭代次数
        verbose: 是否打印优化过程
        """
        # 预处理参考图像
        reference_tensor = self._preprocess_image(reference_image).unsqueeze(0)
        
        # 重置优化器状态
        self.optimizer.zero_grad()
        
        # 开始优化迭代
        for i in range(iterations):
            # 使用当前旋转角度渲染图像
            rendered_image = renderer.render(
                point_cloud,
                angles=self.rotation.detach().cpu().numpy() * 180 / np.pi,  # 转换为度
                resolution=(512, 512)
            )
            
            # 转换为张量
            rendered_tensor = torch.tensor(rendered_image, dtype=torch.float32, 
                                          device=self.device).permute(2, 0, 1) / 255.0
            rendered_tensor = rendered_tensor.unsqueeze(0)
            
            # 计算损失
            loss = self.loss_fn(rendered_tensor, reference_tensor)
            
            # 反向传播和优化
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            
            # 记录损失
            loss_val = loss.item()
            self.loss_history.append(loss_val)
            
            # 更新最佳旋转角度
            if loss_val < self.best_loss:
                self.best_loss = loss_val
                self.best_rotation = self.rotation.detach().cpu().numpy().copy()
            
            # 限制旋转角度范围在 [-90, 90] 度
            with torch.no_grad():
                self.rotation.clamp_(-np.pi/2, np.pi/2)  # 弧度
            
            # 打印进度
            if verbose and (i + 1) % 5 == 0:
                print(f"Iteration {i+1}/{iterations}, Loss: {loss_val:.6f}")
        
        return {
            "best_rotation": self.best_rotation * 180 / np.pi,  # 转换为度
            "best_loss": self.best_loss,
            "loss_history": self.loss_history
        }
    
    def plot_loss_history(self, save_path=None):
        """绘制损失历史曲线"""
        plt.figure(figsize=(10, 5))
        plt.plot(self.loss_history)
        plt.title("Camera Alignment Loss History")
        plt.xlabel("Iteration")
        plt.ylabel("Loss")
        plt.grid(True)
        
        if save_path:
            plt.savefig(save_path)
            print(f"Loss history plot saved to {save_path}")
        else:
            plt.show()
        
        plt.close()

class GaussianRenderer:
    """高斯点云渲染器，用于从3D点云生成2D图像"""
    
    def __init__(self, image_size=(512, 512), device="cuda"):
        self.image_size = image_size
        self.device = device
        self.camera = Camera(device=device)
    
    def render(self, point_cloud, angles=[0.0, 0.0, 0.0], resolution=None):
        """
        渲染3D点云为2D图像
        point_cloud: 高斯点云数据字典
        angles: 相机旋转角度 [x, y, z]（度）
        resolution: 输出图像分辨率
        返回: 渲染的图像，形状为 (H, W, 3)
        """
        # 设置分辨率
        if resolution is not None:
            h, w = resolution
        else:
            h, w = self.image_size
        
        # 设置相机角度
        self.camera.set_pose(rotation=angles)
        
        # 获取点云数据
        points = point_cloud['means3D'][0]  # [N, 3]
        colors = point_cloud['colors'][0]   # [N, 3]
        
        # 投影3D点到2D图像平面
        projected_points = self.camera.project_points(points)  # [N, 2]
        
        # 将标准化设备坐标 (NDC) 转换为像素坐标
        pixel_x = ((projected_points[:, 0] + 1) * 0.5) * w
        pixel_y = ((-projected_points[:, 1] + 1) * 0.5) * h  # 注意y轴反转
        
        # 确保像素坐标在图像范围内
        valid = (pixel_x >= 0) & (pixel_x < w) & (pixel_y >= 0) & (pixel_y < h)
        pixel_x = pixel_x[valid].long()
        pixel_y = pixel_y[valid].long()
        colors = colors[valid]
        
        # 创建空白图像
        image = torch.zeros((h, w, 3), device=self.device)
        depth_buffer = torch.full((h, w), -1e9, device=self.device)  # 用于深度测试
        
        # 计算点的深度（Z坐标，已通过视图变换）
        # 简化实现：使用原始Z坐标作为深度代理
        depths = points[valid, 2]
        
        # 绘制点到图像上，进行深度测试
        for x, y, z, color in zip(pixel_x, pixel_y, depths, colors):
            if z > depth_buffer[y, x]:
                image[y, x] = color
                depth_buffer[y, x] = z
        
        # 转换为numpy图像并调整范围
        image = image.cpu().detach().numpy()
        image = (image * 255).clip(0, 255).astype(np.uint8)
        
        return image
    
    def render_multiple_views(self, point_cloud, angles_list, resolution=None):
        """
        渲染多个视角
        angles_list: 角度列表，每个元素是 [x, y, z] 角度
        返回: 多个视角的图像列表
        """
        views = []
        for i, angles in enumerate(angles_list):
            view = self.render(point_cloud, angles, resolution)
            views.append({
                "image": view,
                "angles": angles
            })
            print(f"Rendered view {i+1}/{len(angles_list)}")
        return views
    
    def save_rendered_image(self, image, save_path):
        """保存渲染的图像"""
        img = Image.fromarray(image)
        img.save(save_path)
        print(f"Rendered image saved to {save_path}")
