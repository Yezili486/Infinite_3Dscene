import torch
import numpy as np
import cv2
from PIL import Image

class ModelWrapper:
    def __init__(self, config):
        self.config = config
        self.device = torch.device(config.device)
        self.models = self._load_models()

    def _load_models(self):
        """加载模型（简化版本）"""
        print("加载模型...")
        return {
            "esrgan": self._create_simple_enhancer(),
            "zoe": self._create_simple_depth_estimator(),
            "closeup_gs": self._create_simple_closeup_gs(),
            "renderer": self._create_simple_renderer()
        }

    def _create_simple_enhancer(self):
        """创建简单的图像增强器"""
        class SimpleEnhancer:
            def enhance(self, img):
                # 简单的图像增强：调整大小和对比度
                img_pil = Image.fromarray(img)
                img_resized = img_pil.resize((img.shape[1]*2, img.shape[0]*2), Image.LANCZOS)
                img_array = np.array(img_resized)
                # 简单的对比度增强
                img_enhanced = np.clip(img_array * 1.2, 0, 255).astype(np.uint8)
                return img_enhanced, None
        return SimpleEnhancer()

    def _create_simple_depth_estimator(self):
        """创建简单的深度估计器"""
        class SimpleDepthEstimator:
            def infer(self, img):
                # 简单的深度估计：基于灰度图的模糊
                gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
                depth = cv2.GaussianBlur(gray, (15, 15), 0)
                return depth
        return SimpleDepthEstimator()

    def _create_simple_closeup_gs(self):
        """创建简单的Closeup GS模型"""
        class SimpleCloseupGS:
            def __call__(self, pc_data):
                # 简单的点云增强
                points = pc_data["points"]
                colors = pc_data["colors"]
                # 添加一些噪声来模拟增强效果
                noise = torch.randn_like(points) * 0.01
                enhanced_points = points + noise
                return {"points": enhanced_points, "colors": colors}
        return SimpleCloseupGS()

    def _create_simple_renderer(self):
        """创建简单的渲染器"""
        class SimpleRenderer:
            def __init__(self):
                self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            
            def __call__(self, pc_data, angle=0, iterations=500):
                # 简单的渲染：将3D点投影到2D
                points = pc_data["points"].cpu().numpy()
                colors = pc_data["colors"].cpu().numpy()
                
                # 简单的3D到2D投影
                h, w = 512, 512
                # 旋转点云
                angle_rad = np.radians(angle)
                cos_a, sin_a = np.cos(angle_rad), np.sin(angle_rad)
                rotation_matrix = np.array([
                    [cos_a, 0, sin_a],
                    [0, 1, 0],
                    [-sin_a, 0, cos_a]
                ])
                
                rotated_points = points @ rotation_matrix.T
                
                # 投影到2D
                x_2d = (rotated_points[:, 0] + 1) * w / 2
                y_2d = (rotated_points[:, 1] + 1) * h / 2
                
                # 创建图像
                img = np.zeros((h, w, 3), dtype=np.uint8)
                valid_mask = (x_2d >= 0) & (x_2d < w) & (y_2d >= 0) & (y_2d < h)
                
                if np.any(valid_mask):
                    x_valid = x_2d[valid_mask].astype(int)
                    y_valid = y_2d[valid_mask].astype(int)
                    colors_valid = colors[valid_mask]
                    
                    # 将颜色值转换到0-255范围
                    colors_255 = (colors_valid * 255).astype(np.uint8)
                    
                    # 在图像上绘制点
                    for i in range(len(x_valid)):
                        if 0 <= x_valid[i] < w and 0 <= y_valid[i] < h:
                            img[y_valid[i], x_valid[i]] = colors_255[i]
                
                return img

        return SimpleRenderer()

    def process(self, input_data):
        results = []
        print("开始处理图像...")
        
        for i, item in enumerate(input_data):
            print(f"处理图像 {i+1}/{len(input_data)}: {item['name']}")
            
            # 1. 图像增强
            sr_img, _ = self.models["esrgan"].enhance(item["data"])
            print(f"  - 图像增强完成")
            
            # 2. 深度估计
            depth = self.models["zoe"].infer(sr_img)
            print(f"  - 深度估计完成")
            
            # 3. 创建点云
            pc = self._create_point_cloud(sr_img, depth)
            print(f"  - 点云生成完成")
            
            # 4. Closeup GS增强
            enhanced_pc = self.models["closeup_gs"](pc)
            print(f"  - Closeup GS增强完成")
            
            # 5. 渲染多角度视图
            renders = []
            for angle in self.config.render_views:
                render = self.models["renderer"](enhanced_pc, angle=angle, iterations=self.config.render_iterations)
                renders.append(render)
            print(f"  - 渲染完成 ({len(renders)} 个视角)")
            
            results.append({"name": item["name"], "renders": renders})
        
        return results

    def _create_point_cloud(self, img, depth):
        h, w = img.shape[:2]
        xx, yy = np.meshgrid(np.arange(w), np.arange(h))
        z = depth / 1000.0
        x3d = (xx - w//2) * z / (w/2)
        y3d = (yy - h//2) * z / (h/2)
        points = np.stack([x3d, y3d, z], axis=-1).reshape(-1, 3)
        colors = img.reshape(-1, 3) / 255.0
        
        # 随机采样点
        if len(points) > self.config.point_cloud_density:
            idx = np.random.choice(len(points), self.config.point_cloud_density, replace=False)
            points = points[idx]
            colors = colors[idx]
        
        return {
            "points": torch.tensor(points, dtype=torch.float32).to(self.device),
            "colors": torch.tensor(colors, dtype=torch.float32).to(self.device)
        } 