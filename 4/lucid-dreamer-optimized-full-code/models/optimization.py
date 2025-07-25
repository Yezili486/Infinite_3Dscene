import torch
import os
from tqdm import tqdm
import logging
from datetime import datetime

class ProgressiveTrainer:
    def __init__(self, config, dreamer, alignment, gaussian_model):
        self.config = config
        self.device = config['device']
        self.dreamer = dreamer
        self.alignment = alignment
        self.model = gaussian_model
        
        # 初始化优化器
        self.optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=2e-4,
            betas=(0.9, 0.999)
        )
        
        # 初始化日志
        self._init_logger()
        
        # 创建输出目录
        os.makedirs(config['paths']['output_dir'], exist_ok=True)
        os.makedirs(config['paths']['log_dir'], exist_ok=True)

    def _init_logger(self):
        """初始化训练日志"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_file = os.path.join(self.config['paths']['log_dir'], f"train_{timestamp}.log")
        
        self.logger = logging.getLogger("ProgressiveTrainer")
        self.logger.setLevel(logging.INFO)
        
        formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        
        # 文件 handler
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(formatter)
        
        # 控制台 handler
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(formatter)
        
        self.logger.addHandler(file_handler)
        self.logger.addHandler(console_handler)

    def set_resolution(self, resolution):
        """设置渲染分辨率"""
        self.model.render_params['width'] = resolution
        self.model.render_params['height'] = resolution
        self.logger.info(f"设置渲染分辨率: {resolution}x{resolution}")

    def train_stage(self, epochs, lr=None):
        """单阶段训练"""
        if lr:
            for param_group in self.optimizer.param_groups:
                param_group['lr'] = lr
            self.logger.info(f"设置学习率: {lr}")
        
        # 获取多视角训练数据
        input_img = cv2.imread(os.path.join(self.config['paths']['input_dir'], "input.jpg"))
        input_tensor = self.dreamer.preprocess_image(input_img)
        views = self.dreamer.generate_multiview(input_tensor)
        
        # 生成相机参数（简化实现）
        camera_params = self._generate_camera_params(len(views))
        
        # 训练循环
        for epoch in tqdm(range(epochs), desc=f"训练阶段 (共{epochs}轮)"):
            total_loss = 0.0
            loss_details = {}
            
            for i, (view, cam_params) in enumerate(zip(views, camera_params)):
                self.optimizer.zero_grad()
                
                # 计算损失
                loss, details = self.alignment.compute_alignment_loss(view, cam_params)
                loss.backward()
                self.optimizer.step()
                
                total_loss += loss.item()
                for k, v in details.items():
                    if k not in loss_details:
                        loss_details[k] = 0.0
                    loss_details[k] += v
            
            # 计算平均损失
            avg_loss = total_loss / len(views)
            for k in loss_details:
                loss_details[k] /= len(views)
            
            # 日志记录
            if (epoch + 1) % 10 == 0:
                self.logger.info(f"轮次 {epoch+1}/{epochs} - 平均损失: {avg_loss:.6f}")
                for k, v in loss_details.items():
                    self.logger.info(f"  {k}: {v:.6f}")

    def train(self):
        """完整训练流程：两阶段渐进式训练"""
        self.logger.info("开始渐进式训练...")
        
        # 阶段1：低分辨率快速拟合
        stage1_res = self.config['model']['progressive_training']['stage1_res']
        stage1_epochs = self.config['model']['progressive_training']['stage1_epochs']
        self.logger.info(f"===== 阶段1: 分辨率 {stage1_res}x{stage1_res}, 共{stage1_epochs}轮 =====")
        self.set_resolution(stage1_res)
        self.train_stage(stage1_epochs, lr=2e-4)
        
        # 阶段2：高分辨率优化细节
        stage2_res = self.config['model']['progressive_training']['stage2_res']
        stage2_epochs = self.config['model']['progressive_training']['stage2_epochs']
        self.logger.info(f"===== 阶段2: 分辨率 {stage2_res}x{stage2_res}, 共{stage2_epochs}轮 =====")
        self.set_resolution(stage2_res)
        self.train_stage(stage2_epochs, lr=1e-4)
        
        # 保存最终模型
        model_path = os.path.join(self.config['paths']['output_dir'], "optimized_3dgs.pth")
        torch.save(self.model.state_dict(), model_path)
        self.logger.info(f"训练完成，模型已保存至: {model_path}")

    def _generate_camera_params(self, num_views):
        """生成多视角相机参数（简化实现）"""
        # 实际应用中应使用真实的相机内外参
        camera_params = []
        for i in range(num_views):
            angle = (i / num_views) * 360
            # 简化的相机参数：位置、旋转、内参
            params = {
                'position': [3.0 * torch.sin(torch.tensor(angle * torch.pi / 180)), 0, 3.0 * torch.cos(torch.tensor(angle * torch.pi / 180))],
                'rotation': [0, angle, 0],
                'focal': 1000.0,
                'cx': self.model.render_params['width'] / 2,
                'cy': self.model.render_params['height'] / 2
            }
            camera_params.append(params)
        return camera_params
